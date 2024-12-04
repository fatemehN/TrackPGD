from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os, cv2
import numpy as np
import torch, math

import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms.functional as tvisf

CUDA_LAUNCH_BLOCKING=1
torch.set_grad_enabled(True)


def where(cond, x, y):
    """
    code from :
        https://discuss.pytorch.org/t/how-can-i-do-the-operation-the-same-as-np-where/1329/8
    """
    cond = cond.float()
    return (cond*x) + ((1-cond)*y)


def sample_target(im, target_bb, search_area_factor, output_sz=256, mode=cv2.BORDER_REPLICATE):
    """ Extracts a crop centered at target_bb box, of size search_area_factor times target_bb(Both height and width)

    args:
        im - cv image
        target_bb - target box [x, y, w, h]
        search_area_factor - Ratio of crop size to target size
        output_sz - (float) Size to which the extracted crop is resized (always square). If None, no resizing is done.

    returns:
        cv image - extracted crop
        float - the factor by which the crop has been resized to make the crop size equal output_size
    """
    #im = im.clone().detach().cpu().numpy()
    x, y, w, h = target_bb.tolist()

    # Crop image
    ws = math.ceil(search_area_factor * w)
    hs = math.ceil(search_area_factor * h)

    if ws < 1 or hs < 1:
        return np.zeros((output_sz, output_sz))
        #raise Exception('Too small bounding box.')

    x1 = round(x + 0.5*w - ws*0.5)
    x2 = x1 + ws

    y1 = round(y + 0.5 * h - hs * 0.5)
    y2 = y1 + hs

    x1_pad = max(0, -x1)
    x2_pad = max(x2-im.shape[1]+1, 0)

    y1_pad = max(0, -y1)
    y2_pad = max(y2-im.shape[0]+1, 0)

    # Crop target
    im_crop = im[y1+y1_pad:y2-y2_pad, x1+x1_pad:x2-x2_pad]

    # Pad
    im_crop_padded = cv2.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, mode)


    im_crop_padded_rsz = cv2.resize(im_crop_padded, (output_sz, output_sz))
    
    return im_crop_padded_rsz #, h_rsz_f, w_rsz_f
    
 
def get_cls_loss(pred, label, select):
    if len(select.size()) == 0 or \
            select.size() == torch.Size([0]):
        return 0
    pred = torch.index_select(pred, 0, select)
    label = torch.index_select(label, 0, select)
    return F.nll_loss(pred, label.long())

def select_cross_entropy_loss(msk, target):
    pred = msk.view(-1, 2)
    label = target.view(-1)
    obj = label.data.eq(1).nonzero().squeeze()
    bcg = label.data.eq(0).nonzero().squeeze()
    loss_pos = get_cls_loss(pred, label, obj)
    loss_neg = get_cls_loss(pred, label, bcg)
    return loss_pos, loss_neg



def delta2bbox(delta):
    bbox_cxcywh = delta.clone()
    '''based on (128,128) center region'''
    bbox_cxcywh[:, :2] = 128.0 + delta[:, :2] * 128.0  # center offset
    bbox_cxcywh[:, 2:] = 128.0 * torch.exp(delta[:, 2:])  # wh revise
    bbox_xywh = bbox_cxcywh.clone()
    bbox_xywh[:, :2] = bbox_cxcywh[:, :2] - 0.5 * bbox_cxcywh[:, 2:]
    return bbox_xywh



def img_preprocess(img_arr):
    '''---> Pytorch tensor(RGB),Normal(-1 to 1,subtract mean, divide std)
    input img_arr (H,W,3)
    output (1,1,3,H,W)
    '''
    mean = np.array([0.485, 0.456, 0.406]).reshape((1,1,3))
    std = np.array([0.229, 0.224, 0.225]).reshape((1,1,3))
    norm_img = ((img_arr/255.0) - mean)/(std)
    img_f32 = norm_img.astype(np.float32)
    img_tensor = torch.from_numpy(img_f32).cuda()
    img_tensor = img_tensor.permute((2,0,1))
    return img_tensor.unsqueeze(dim=0).unsqueeze(dim=0)


def img_preprocessT(img):
    '''---> Pytorch tensor(RGB),Normal(-1 to 1,subtract mean, divide std)
    input img_arr (H,W,3)
    output (1,1,3,H,W)
    '''
    im = img.squeeze(dim=0).squeeze(dim=0).permute((1, 2, 0))
    image = im.data.to(device='cuda', dtype=torch.float)
    mean = torch.from_numpy(np.array([0.485, 0.456, 0.406]).reshape((1,1,3))).to(device='cuda', dtype=torch.float)
    std = torch.from_numpy(np.array([0.229, 0.224, 0.225]).reshape((1,1,3))).to(device='cuda', dtype=torch.float)
    norm_img = ((image/255.0) - mean)/(std)
    img_tensor = norm_img.permute((2,0,1))
    return img_tensor.unsqueeze(dim=0).unsqueeze(dim=0)


def dice_loss(inputs, targets):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    # print(inputs.max(), inputs.min())
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() 



def sigmoid_focal_loss(inputs, targets, ce_loss, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    # ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() 


def TrackPGD(net, img, bbox, target, dtm, trt, search_factor, epsilon_ = 10, alpha_= 1, max_t=10):  
    """
    TrackPGD attack for OSTrackSTS tracker
    Args:
        net: AlphaRefine network (segmentation head of OSTrackSTS)
        img: Current frame
        target: The previous predicted mask used as the ground truth for computing SegPGD loss
        template_feats: template deep features 
        roi_bb: region of interest (bounding box)
        search_factor: The scale factor of the search region
        epsilon_: The pixel value of \epsilon parameter in TrackPGD
        alpha_: The step size of gradient update
        max_t: The iteration number
    Returns:
        Adversarial search region 
    """
    img = torch.from_numpy(img.astype(np.float32)).cuda()
    img = img.permute((2,0,1)).unsqueeze(dim=0).unsqueeze(dim=0)
    image_init = img
    for t in range(max_t):

        #Normalize for Alpha-refine Network
        x_adv = img_preprocessT(img)
        x_adv = Variable(x_adv.data, requires_grad=True)
        #Segment
        net = net.cuda()
        if dtm is not None:
            pred = net.forward_test(x_adv, dtm, trt, mode='mask')
        else:
            pred = net.forward_test(x_adv, trt=trt, mode='mask')   
        msk = pred.squeeze(dim=0).squeeze(dim=0) #.permute((1, 2, 0))

        #####Compute the SegPGD loss for foreground mask
        #Create the one-hot encoded mask for cross entropy loss function
        encoded_msk = torch.stack((1 - msk, msk), 2)
        #Crop Target mask and make it one-hot 
        gt = sample_target(target, bbox, search_factor, output_sz=384, mode=cv2.BORDER_REPLICATE)
        groudnTruth = torch.from_numpy(gt.astype(np.float32)).cuda()
        #Compute the weighted BCE loss
        loss_pos, loss_neg = select_cross_entropy_loss(encoded_msk.float().contiguous(), groudnTruth.contiguous())

        #Compute Lambda
        lmbd = t / 2 / max_t
        #Compute the SegPGD loss for foreground
        loss_fg = (1 - lmbd) * loss_pos + lmbd * loss_neg

        #####Compute the SegPGD for background mask
        #Create background ground truth (1- G_\tau)
        alt_groundTruth = 1 - groudnTruth
        #Compute the weighted BCE loss
        loss_pos_alt, loss_neg_alt = select_cross_entropy_loss(encoded_msk.float().contiguous(), alt_groundTruth.contiguous())

        #Compute the SegPGD loss for background
        loss_bg = (1 - lmbd) * loss_pos_alt + lmbd * loss_neg_alt
        #Compute difference loss L_\Delta
        loss_cls = loss_fg - loss_bg 
        #Compute the focal loss via L_\Delta
        loss_focal = sigmoid_focal_loss(msk, groudnTruth, loss_cls)
        loss_dice = dice_loss(msk, groudnTruth)

        #Compute TrackPGD loss
        loss = 10*loss_dice + 1*loss_focal
        # calculate the derivative
        net.zero_grad()
        loss.backward(retain_graph=True)

        #Compute Perturbation
        adv_grad = x_adv.grad
        adv_grad = torch.sign(adv_grad) 
        pert = alpha_ * adv_grad 

        #Generate the pertubed frame
        img = img + pert 
        img = where(img > image_init + epsilon_, image_init + epsilon_, img)
        img = where(img < image_init - epsilon_, image_init - epsilon_ , img)
        x_adv.data = torch.clamp(img, 0, 255)
        img = x_adv.data
    
    img_adv = x_adv.squeeze(dim=0).squeeze(dim=0).permute((1, 2, 0))
    return img_adv.detach().cpu().numpy()