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

download and move to the proper folder:
// https://github.com/open-mmlab/mmaction2/blob/main/tools/data/skeleton/ntu_pose_extraction.py
	modify lines:
	15// args.det_config = '../../../demo/demo_configs/faster-rcnn_r50-caffe_fpn_ms-1x_coco-person.py'  # noqa: E501
	18// args.pose_config = '../../../demo/demo_configs/td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py'  # noqa: E501
// https://github.com/open-mmlab/mmaction2/blob/main/tools/data/skeleton/S001C001P001R001A001_rgb.avi
// https://github.com/open-mmlab/mmaction2/blob/main/demo/demo_configs/faster-rcnn_r50-caffe_fpn_ms-1x_coco-person.py #POSE DETECTION, file info https://github.com/open-mmlab/mmdetection/tree/main/configs/faster_rcnn
// https://github.com/open-mmlab/mmaction2/blob/main/demo/demo_configs/td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py #POSE ESTIMATION, file info https://github.com/open-mmlab/mmpose/tree/master/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco checkpoint sama dengan hrnet_w32_coco_256x192.py

run ntu_pose_extraction.py:
// python ntu_pose_extraction.py S001C001P001R001A001_rgb.avi S001C001P001R001A001_rgb.pkl

Neural Network Architecture Used:
- faster-rcnn_r50-caffe_fpn_ms-1x_coco-person
	//checkpoint = https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco-person/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth
- td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer
	//checkpoint = https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth
