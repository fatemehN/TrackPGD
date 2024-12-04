# TrackPGD
[NeurIPS 2024 in AdvML Workshop]  TrackPGD: Efficient Adversarial Attack using Object Binary Masks against Robust Transformer Trackers

[Project Webpage](https://lvsn.github.io/TrackPGD/)
[OpenReview] coming soon!
[ArXiv](https://arxiv.org/abs/2407.03946)

### TrackPGD Attack for MixFormerM and OSTrackSTS trackers

## Mask Evaluation 
## Step1: Download the trackers packages
Please download the trackers from the VOT challenge (VOT2022) website, as follows:
1- MixFormerM: http://data.votchallenge.net/vot2022/trackers/MixFormerM-code-2022-05-04T09_55_58.619853.zip
2- OSTrackSTS: https://data.votchallenge.net/vot2022/trackers/OSTrackSTS-code-2022-05-03T18_41_55.435766.zip 

## Step2: Create the environment
For each tracker follow the instructions to build the suitable environment as stated in their README.md file. 

## Step3: Download the networks 
For TrackPGD experiments, we used the following networks:

1- MixFormerM: 
    1.1- Tracker network(mixformer_vit_score_imagemae.pth.tar) from https://drive.google.com/file/d/1EOZgd3HVlTmhPdsWd-zGqx4I53H4oiqf/view 
    Place this network on the (MixFormerM_submit/mixformer/models)
    1.2- Segmentation network(SEcmnet_ep0440.pth.tar) from https://drive.google.com/file/d/1J0ebV0Ksye62yQOba8ymCoWFFg-MxXVy/view
    Place this network on the (MixFormerM_submit/mixformer/external/AR/ltr/checkpoints/ltr/ARcm_seg/ARcm_coco_seg_only_mask_384)

2- OSTrackSTS:
    1.1- Tracker network(ostrack320_elimination_cls_t2m12_seg_ep50-20230706T225239Z-001.zip): from https://drive.google.com/drive/folders/1PwG4i25GZFsB8g5W0E-tZUMUSUlVzcCz?usp=sharing
    Download the pre-trained weight under the folder "ostrack320_elimination_cls_t2m12_seg_ep50" and place the weight file under
    "$PROJ_ROOT$/output/checkpoints/train/ostrack/ostrack320_elimination_cls_t2m12_seg_ep50"
    1.2- Segmentation network(baseline_plus_got_lasot): from https://drive.google.com/drive/folders/1PwG4i25GZFsB8g5W0E-tZUMUSUlVzcCz?usp=sharing 
    Besides, you may also need to download the weights of AlphaRefine under the folder "baseline_plus_got_lasot" and place the weight file under
    "$PROJ_ROOT$/external/AR_VOT22/checkpoints/ltr/ARcm_seg/baseline_plus_got_lasot" 


## Step 4: Run the setup files 
Follow the instructions of each tracker to correct the paths and run the setup files. 

## Step 5.a: Set the paths for MixFormerM
1- From MixFormerM folder on TrackPGD directory(TrackPGD_AAAI/code/MixFormerM), find TrackPGD folder. Copy and paste this folder into the tracker folder (/MixFormerM_submit/mixformer/external/AR/pytracking/). 

2- Add a new entry to trackers.ini file in "vot22_seg_mixformer_large" directory(/MixFormerM_submit/mixformer/vot22_seg_mixformer_large) as follows:
```
######################################
[MixFormer_TrackPGD]  
label = MixFormer_TrackPGD
protocol = traxpython
command = mixformer_vit_large_vit_seg_class_TrackPGD
paths = <PATH_OF_MIXFORMER>:<PATH_OF_MIXFORMER>/external/AR/pytracking/TrackPGD:<PATH_OF_MIXFORMER>/external/AR
env_PATH = <PATH_OF_PYTHON>
#####################################
```

3- Edit the paths of MixFormer_TrackPGD entry to include all of the necessary paths as recommended on the tracker' README.md file. The <PATH_OF_MIXFORMER> is your path to "mixformer" folder.

4- Edit <PATH_OF_PYTHON> with your path to the MixFormer environment you built in ##Step 2.

## Step 5.b: Set the paths for OSTrackSTS
1- From OSTrackSTS folder on TrackPGD directory(TrackPGD_AAAI/code/OSTrackSTS), find TrackPGD folder. Copy and paste this folder to the (OSTrack/external/AR_VOT22/pytracking) of the tracker folder.

2- Add a new entry to trackers.ini file in "vot22/OSTrackSTS" directory(OStrack/external/vot22/OSTrackSTS) as follows:

```
####################################
[OSTrackSTS_TrackPGD]  
label = OSTrackSTS_TrackPGD
protocol = traxpython
command = OSTrackSTS_TrackPGD
paths = <PATH_OF_OSTrack>:<PATH_OF_OSTrack>//external/AR_VOT22/pytracking/TrackPGD
env_PATH = <PATH_OF_PYTHON>
####################################
```

3- Edit the paths of the OSTrackSTS_TrackPGD entry to include all of the necessary paths as recommended on the tracker' README.md file.  The <PATH_OF_OSTrack> is your path to the "OSTrack" folder.

4- Edit <PATH_OF_PYTHON> with your path to the OSTrack environment you built in ##Step 2.


## Step 6.a: Run the MixFormerM tracker attacked by TrackPGD for VOT2022STS evaluation
1- Enter the VOT workplace directory (/path/to/vot22_seg_mixformer_large)
2- Activate the MixFormer environment. 
3- Run:
```
vot evaluate --workspace . MixFormer_TrackPGD
vot analysis --workspace .
```


## Step 6.b: Run the OSTrackSTS tracker attacked by TrackPGD for VOT2022STS evaluation
1- Enter the VOT workspace directory (/path/to/vot22/OSTrackSTS/)
2- Activate the OSTrack environment. 
3- Run:
```
vot evaluate --workspace . OSTrackSTS_TrackPGD
vot analysis --workspace .
```


## Contact:

[Fatemeh Nokabadi](mailto:nourifatemeh1@gmail.com)
