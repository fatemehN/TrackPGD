import sys
import os
filepath = os.path.abspath(__file__)
AR_dir = os.path.join(os.path.dirname(filepath), "..", "..")
sys.path.append(AR_dir)
stark_dir = os.path.join(AR_dir, "..", "..")
stark_lib_dir = os.path.join(stark_dir, "lib")
sys.path.append(stark_dir)
sys.path.append(stark_lib_dir)

from pytracking.VOT2022.stark_alpha_seg_class_TrackPGD import run_vot_exp
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
run_vot_exp('ostrack_online', 'ostrack320_elimination_cls_t2m12_seg_ep50', 'baseline_plus_got_lasot', 0.40, VIS=False, trt='false')
