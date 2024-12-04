import os
import sys
import torch
import numpy as np
import cv2
import torch.nn as nn
from external.AR.pytracking.utils.loading import load_network
from external.AR.ltr.data.processing_utils_SE import sample_target_SE, transform_image_to_crop_SE, map_mask_back

import TrackPGD_C1 as utls

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)


def delta2bbox(delta):
    bbox_cxcywh = delta.clone()
    '''based on (128,128) center region'''
    bbox_cxcywh[:, :2] = 128.0 + delta[:, :2] * 128.0  # center offset
    bbox_cxcywh[:, 2:] = 128.0 * torch.exp(delta[:, 2:])  # wh revise
    bbox_xywh = bbox_cxcywh.clone()
    bbox_xywh[:, :2] = bbox_cxcywh[:, :2] - 0.5 * bbox_cxcywh[:, 2:]
    return bbox_xywh


class ARcm_seg(object):
    def __init__(self, refine_net_dir, search_factor=2.0, input_sz=256):
        self.refine_network = self.get_network(refine_net_dir)
        self.search_factor = search_factor
        self.input_sz = input_sz
        self.mean = np.array([0.485, 0.456, 0.406]).reshape((1,1,3))
        self.std = np.array([0.229, 0.224, 0.225]).reshape((1,1,3))

    def initialize(self, frame1, bbox1):
        '''
        :param frame1: cv array (H,W,3)
        :param bbox1: ndarray (4,)
        :return:
        '''
        '''Step1: get cropped patch(tensor)'''
        patch1, h_f, w_f = sample_target_SE(frame1, bbox1, self.search_factor, self.input_sz, mode=cv2.BORDER_CONSTANT)
        patch1_tensor = self.img_preprocess(patch1)
        '''Step2: get GT's cooridinate on the cropped patch(tensor)'''
        crop_sz = torch.Tensor((self.input_sz, self.input_sz))
        bbox1_tensor = self.gt_preprocess(bbox1) # (4,)
        bbox1_crop_tensor = transform_image_to_crop_SE(bbox1_tensor, bbox1_tensor, h_f, w_f, crop_sz).cuda()
        '''Step3: forward prop (reference branch)'''
        with torch.no_grad():
            _, self.template_feats, self.roi_bb = self.refine_network.forward_ref(patch1_tensor, bbox1_crop_tensor)

    '''refine'''
    def get_mask(self, Cframe, Cbbox, prev_msk, dtm=None, vis=False):
        '''
        :param Cframe: Current frame(cv2 array)
        :param Cbbox: Current bbox (ndarray) (x1,y1,w,h)
        :return: mask
        '''
        '''Step1: get cropped patch(tensor)'''
        Cpatch, h_f, w_f = sample_target_SE(Cframe, Cbbox, self.search_factor, self.input_sz, mode=cv2.BORDER_CONSTANT)
        '''
        Apply TrackPGD Attack
        '''
        img_adv = utls.TrackPGD(self.refine_network, Cpatch, prev_msk, self.template_feats,self.roi_bb, self.search_factor)
        
        Cpatch_tensor = self.img_preprocess(img_adv)
        '''Step2: forward prop (test branch)'''
        # with torch.no_grad():
        if dtm is not None:
            '''2020.4.26 support input dtm'''
            # pred = self.refine_network.forward_test(Cpatch_tensor, dtm, mode='mask')
            pred = self.refine_network.forward_test(Cpatch_tensor, self.template_feats, self.roi_bb, mode='test')
        else:
            # pred = self.refine_network.forward_test(Cpatch_tensor,mode='mask')
            pred = self.refine_network.forward_test(Cpatch_tensor,self.template_feats, self.roi_bb, mode='test')
            # pred = pred['mask']
        # Pmask_arr = mask_torch2numpy(pred['mask'])
        Pmask_arr = self.pred2bbox(pred, input_type='mask')
        mask_arr = map_mask_back(Cframe, Cbbox, self.search_factor, Pmask_arr,
                                    mode=cv2.BORDER_CONSTANT)
        
        Pbbox_arr = self.pred2bbox(pred, input_type='corner')
        bbox_arr = self.bbox_back(Pbbox_arr, Cbbox, h_f, w_f)
        
        # if vis:
        #     return mask_arr, Cpatch, Pmask_arr
        # else:
        #     return mask_arr
        return mask_arr, bbox_arr, img_adv, Cpatch

    
    def pred2bbox(self, prediction, input_type=None):
        if input_type == 'bbox':
            Pbbox = prediction['bbox']
            Pbbox = delta2bbox(Pbbox)
            Pbbox_arr = np.array(Pbbox.squeeze().cpu())
            return Pbbox_arr

        elif input_type == 'corner':
            Pcorner = prediction['corner']  # (x1,y1,x2,y2)
            # Pbbox_arr = np.array(Pcorner.squeeze().cpu())
            Pbbox_arr = np.array(Pcorner.squeeze().detach().cpu().numpy())
            Pbbox_arr[2:] = Pbbox_arr[2:] - Pbbox_arr[:2]  # (x1,y1,w,h)
            return Pbbox_arr

        elif input_type == 'mask':
            Pmask = prediction['mask']
            Pmask_arr = np.array(Pmask.squeeze().detach().cpu())  # (H,W) (0,1)
            return Pmask_arr

        else:
            raise ValueError("input_type should be 'bbox' or 'mask' or 'corner' ")


    def bbox_back(self, bbox_crop, bbox_ori, h_f, w_f):
        """
        Args:
            bbox_crop: coordinate on (256x256) region in format (x1,y1,w,h) (4,)
            bbox_ori: origin traking result (x1,y1,w,h) (4,)
            h_f: h scale factor
            w_f: w scale factor
        Return:
            coordinate mapping back to origin image
        """
        x1_c, y1_c, w_c, h_c = bbox_crop.tolist()
        x1_o, y1_o, w_o, h_o = bbox_ori.tolist()
        x1_oo = x1_o - (self.search_factor-1)/2 * w_o
        y1_oo = y1_o - (self.search_factor-1)/2 * h_o
        delta_x1 = x1_c / w_f
        delta_y1 = y1_c / h_f
        delta_w = w_c / w_f
        delta_h = h_c / h_f
        return np.array([x1_oo + delta_x1, y1_oo + delta_y1,
                         delta_w, delta_h])



    def get_network(self,checkpoint_dir):
        network = load_network(checkpoint_dir)
        network.cuda()
        network.eval()
        return network

    def img_preprocess(self,img_arr):
        '''---> Pytorch tensor(RGB),Normal(-1 to 1,subtract mean, divide std)
        input img_arr (H,W,3)
        output (1,1,3,H,W)
        '''
        norm_img = ((img_arr/255.0) - self.mean)/(self.std)
        img_f32 = norm_img.astype(np.float32)
        img_tensor = torch.from_numpy(img_f32).cuda()
        img_tensor = img_tensor.permute((2,0,1))
        return img_tensor.unsqueeze(dim=0).unsqueeze(dim=0)

    def gt_preprocess(self,gt_arr):
        '''
        :param gt: ndarray (4,)
        :return: torch tensor (4,)
        '''
        return torch.from_numpy(gt_arr.astype(np.float32))


def add_frame_mask(frame, mask, threshold=0.5):
    mask_new = (mask>threshold)*255 #(H,W)
    frame_new = frame.copy().astype(np.float)
    frame_new[...,1] += 0.3*mask_new
    frame_new = frame_new.clip(0,255).astype(np.uint8)
    return frame_new


def add_frame_bbox(frame, refined_box, color):
    x1, y1, w, h = refined_box.tolist()
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)
    return frame
