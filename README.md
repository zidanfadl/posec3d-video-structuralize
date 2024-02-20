conda:
// mkdir -p ~/miniconda3
// wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
// bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
// rm -rf ~/miniconda3/miniconda.sh
then
// ~/miniconda3/bin/conda init bash
// ~/miniconda3/bin/conda init zsh

env openmmlab:
// conda create --name openmmlab python=3.8 -y
// conda activate openmmlab

conda pytorch:
// conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

test cuda:
// python #entering python terminal:
	// import torch
	// torch.cuda.is_available() #check if cuda enabled and accessible by PyTorch
	// exit()

cuda:
// wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
// sudo dpkg -i cuda-keyring_1.1-1_all.deb
// sudo apt-get update
// sudo apt-get -y install cuda-toolkit-12-3
then
// sudo apt-get install -y cuda-drivers #Enroll MOK

	if missed Enroll MOK screen:
	// sudo mokutil -t /var/lib/shim-signed/mok/MOK.der #check if MOK enrolled
	// sudo mokutil -import /var/lib/shim-signed/mok/MOK.der
	// sudo mokutil -t /var/lib/shim-signed/mok/MOK.der #verify that MOK enrolled
	// sudo reboot now #Enroll MOK, continue, enter password

test cuda:
// python #entering python terminal:
	// import torch
	// torch.cuda.is_available() #verify that cuda enabled and accessible by PyTorch
	// exit()

install:
// pip install -U openmim
// mim install mmengine
// mim install mmcv
// mim install mmdet
// mim install mmpose
// pip install mmaction2

download and move to the proper folder:
// https://github.com/open-mmlab/mmaction2/blob/main/tools/data/skeleton/ntu_pose_extraction.py
	modify lines:
	15// args.det_config = '../../../demo/demo_configs/faster-rcnn_r50-caffe_fpn_ms-1x_coco-person.py'  # noqa: E501
	18// args.pose_config = '../../../demo/demo_configs/td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py'  # noqa: E501
// https://github.com/open-mmlab/mmaction2/blob/main/tools/data/skeleton/S001C001P001R001A001_rgb.avi
// https://github.com/open-mmlab/mmaction2/blob/main/demo/demo_configs/faster-rcnn_r50-caffe_fpn_ms-1x_coco-person.py #POSE DETECTION, file info https://github.com/open-mmlab/mmdetection/tree/main/configs/faster_rcnn
// https://github.com/open-mmlab/mmaction2/blob/main/demo/demo_configs/td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py #POSE ESTIMATION, file info https://github.com/open-mmlab/mmpose/tree/master/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap

run ntu_pose_extraction.py:
// python ntu_pose_extraction.py S001C001P001R001A001_rgb.avi S001C001P001R001A001_rgb.pkl

========================================
Neural Network Architecture Used in ntu_pose_extraction.py:
# HUMAN DETECTION
- mmaction2/demo/demo_configs/faster-rcnn_r50-caffe_fpn_ms-1x_coco-person.py
	//checkpoint = https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco-person/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth
# POSE ESTIMATION
- mmaction2/demo/demo_configs/td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py
	//checkpoint = https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth
	
notes:
One only thing you may need to change is that: since ntu_pose_extraction.py is developed specifically for pose extraction of NTU videos, you can skip the ntu_det_postproc step when using this script for extracting pose from your custom video datasets.
https://github.com/open-mmlab/mmaction2/blob/main/configs/skeleton/posec3d/custom_dataset_training.md
########################################

vlc:
// sudo apt install vlc #optional

testing:
// pip install notebook
// pip install ipython
// pip install moviepy

download and move to the proper folder:
// https://github.com/open-mmlab/mmaction2/blob/main/configs/_base_/default_runtime.py
// https://github.com/open-mmlab/mmaction2/blob/main/tools/data/skeleton/label_map_ntu60.txt
// https://github.com/open-mmlab/mmaction2/blob/main/demo/demo_configs/faster-rcnn_r50_fpn_2x_coco_infer.py #POSE DETECTION, file info https://github.com/open-mmlab/mmdetection/tree/main/configs/faster_rcnn

add drn:
// pip show mmaction2 #remember the location
	download folder:
	// https://github.com/open-mmlab/mmaction2/tree/main/mmaction/models/localizers/drn
	add folder to location:
	// https://github.com/open-mmlab/mmaction2/issues/2714

chrome:
// wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
// sudo dpkg -i google-chrome-stable_current_amd64.deb
// sudo apt-get install -f

run demo_testing.ipynb

========================================
Neural Network Architecture Used in demo_testing.ipynb:
# HUMAN DETECTION
- mmaction2/demo/demo_configs/faster-rcnn_r50_fpn_2x_coco_infer.py
	//checkpoint = http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth
# POSE ESTIMATION
- mmaction2/demo/demo_configs/td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py
	//checkpoint = https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth
# ACTION CLASSIFICATION
- mmaction2/configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py
	//checkpoint = https://download.openmmlab.com/mmaction/skeleton/posec3d/slowonly_r50_u48_240e_ntu60_xsub_keypoint/slowonly_r50_u48_240e_ntu60_xsub_keypoint-f3adabf1.pth
	
visualize heatmap volume: https://github.com/open-mmlab/mmaction2/blob/4e50a824d3abb708619978de65a30eee2daf81bd/demo/visualize_heatmap_volume.ipynb
########################################

download and move to the proper folder:
// https://github.com/open-mmlab/mmaction2/blob/main/tools/train.py
// https://download.openmmlab.com/mmaction/v1.0/skeleton/data/ntu60_2d.pkl #training data at data/skeleton/ntu60_2d.pkl

!python mmaction2/tools/train.py mmaction2/configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py \
    --work-dir work_dirs/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint \
    --seed 0

out of memory solved by modify lines (optional):
mmaction2/configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py
	84// batch_size=4,
	97// batch_size=4,

run demo_training.ipynb

========================================
Neural Network Architecture Used in demo_training.ipynb:
# HUMAN DETECTION & POSE ESTIMATION
- data/skeleton/ntu60_2d.pkl #stated at mmaction2/configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py

# ACTION CLASSIFICATION
- mmaction2/configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py
	//checkpoint = --resume, If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.
########################################

# custom dataset
- pose_extraction.py #skip ntu_det_postproc step
- place pkl at data/skeleton/
- configure ann_file, pretrained, num_classes inside config file, slowonly_..._.py #

##to do
- mmaction2/demo/visualize_heatmap_volume.py
- where is the .pth file saved after training? ~work_dirs/
- add hash_id to .pth file using publish_model.py
- demo_video_structuralize.py
- training video structuralize
