conda:<br />
// mkdir -p ~/miniconda3<br />
// wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh<br />
// bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3<br />
// rm -rf ~/miniconda3/miniconda.sh<br />
then<br />
// ~/miniconda3/bin/conda init bash<br />
// ~/miniconda3/bin/conda init zsh<br />
<br />
env openmmlab:<br />
// conda create --name openmmlab python=3.8 -y<br />
// conda activate openmmlab<br />
<br />
conda pytorch:<br />
// conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia<br />
<br />
test cuda:<br />
// python #entering python terminal:<br />
	// import torch<br />
	// torch.cuda.is_available() #check if cuda enabled and accessible by PyTorch<br />
	// exit()<br />
<br />
cuda:<br />
// wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb<br />
// sudo dpkg -i cuda-keyring_1.1-1_all.deb<br />
// sudo apt-get update<br />
// sudo apt-get -y install cuda-toolkit-12-3<br />
then<br />
// sudo apt-get install -y cuda-drivers #Enroll MOK<br />
<br />
	if missed Enroll MOK screen:<br />
	// sudo mokutil -t /var/lib/shim-signed/mok/MOK.der #check if MOK enrolled<br />
	// sudo mokutil -import /var/lib/shim-signed/mok/MOK.der<br />
	// sudo mokutil -t /var/lib/shim-signed/mok/MOK.der #verify that MOK enrolled<br />
	// sudo reboot now #Enroll MOK, continue, enter password<br />
<br />
test cuda:<br />
// python #entering python terminal:<br />
	// import torch<br />
	// torch.cuda.is_available() #verify that cuda enabled and accessible by PyTorch<br />
	// exit()<br />
<br />
install:<br />
// pip install -U openmim<br />
// mim install mmengine<br />
// mim install mmcv<br />
// mim install mmdet<br />
// mim install mmpose<br />
// pip install mmaction2<br />
<br />
download and move to the proper folder:<br />
// wget https://github.com/open-mmlab/mmaction2/blob/main/tools/data/skeleton/ntu_pose_extraction.py<br />
	modify lines:<br />
	// 15// args.det_config = '../../../demo/demo_configs/faster-rcnn_r50-caffe_fpn_ms-1x_coco-person.py'  # noqa: E501<br />
	// 18// args.pose_config = '../../../demo/demo_configs/td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py'  # noqa: E501<br />
// wget https://github.com/open-mmlab/mmaction2/blob/main/tools/data/skeleton/S001C001P001R001A001_rgb.avi<br />
// wget https://github.com/open-mmlab/mmaction2/blob/main/demo/demo_configs/faster-rcnn_r50-caffe_fpn_ms-1x_coco-person.py #POSE DETECTION, file info https://github.com/open-mmlab/mmdetection/tree/main/configs/faster_rcnn<br />
// wget https://github.com/open-mmlab/mmaction2/blob/main/demo/demo_configs/td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py #POSE ESTIMATION, file info https://github.com/open-mmlab/mmpose/tree/master/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap<br />
<br />
run ntu_pose_extraction.py:<br />
// python ntu_pose_extraction.py S001C001P001R001A001_rgb.avi S001C001P001R001A001_rgb.pkl<br />
<br />

========================================<br />
Neural Network Architecture Used in ntu_pose_extraction.py:<br />
### HUMAN DETECTION<br />
- mmaction2/demo/demo_configs/faster-rcnn_r50-caffe_fpn_ms-1x_coco-person.py<br />
	//checkpoint = https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco-person/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth<br />
### POSE ESTIMATION<br />
- mmaction2/demo/demo_configs/td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py<br />
	//checkpoint = https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth<br />
<br />	
notes:<br />
One only thing you may need to change is that: since ntu_pose_extraction.py is developed specifically for pose extraction of NTU videos, you can skip the ntu_det_postproc step when using this script for extracting pose from your custom video datasets.<br />
https://github.com/open-mmlab/mmaction2/blob/main/configs/skeleton/posec3d/custom_dataset_training.md<br />
########################################

vlc:<br />
// sudo apt install vlc #optional<br />
<br />
testing:<br />
// pip install notebook<br />
// pip install ipython<br />
// pip install moviepy<br />
<br />
download and move to the proper folder:<br />
// wget https://github.com/open-mmlab/mmaction2/blob/main/configs/_base_/default_runtime.py<br />
// wget https://github.com/open-mmlab/mmaction2/blob/main/tools/data/skeleton/label_map_ntu60.txt<br />
// wget https://github.com/open-mmlab/mmaction2/blob/main/demo/demo_configs/faster-rcnn_r50_fpn_2x_coco_infer.py #POSE DETECTION, file info https://github.com/open-mmlab/mmdetection/tree/main/configs/faster_rcnn<br />
<br />
add drn:<br />
// pip show mmaction2 #remember the location<br />
	download folder:<br />
	// https://github.com/open-mmlab/mmaction2/tree/main/mmaction/models/localizers/drn<br />
	add folder to location:<br />
	// https://github.com/open-mmlab/mmaction2/issues/2714<br />
<br />
chrome:<br />
// wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb<br />
// sudo dpkg -i google-chrome-stable_current_amd64.deb<br />
// sudo apt-get install -f<br />
<br />
run demo_testing.ipynb

========================================<br />
Neural Network Architecture Used in demo_testing.ipynb:<br />
### HUMAN DETECTION<br />
- mmaction2/demo/demo_configs/faster-rcnn_r50_fpn_2x_coco_infer.py<br />
	//checkpoint = http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth<br />
### POSE ESTIMATION<br />
- mmaction2/demo/demo_configs/td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py<br />
	//checkpoint = https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth<br />
### ACTION CLASSIFICATION<br />
- mmaction2/configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py<br />
	//checkpoint = https://download.openmmlab.com/mmaction/skeleton/posec3d/slowonly_r50_u48_240e_ntu60_xsub_keypoint/slowonly_r50_u48_240e_ntu60_xsub_keypoint-f3adabf1.pth<br />
<br />
visualize heatmap volume: https://github.com/open-mmlab/mmaction2/blob/4e50a824d3abb708619978de65a30eee2daf81bd/demo/visualize_heatmap_volume.ipynb<br />
########################################

download and move to the proper folder:<br />
// wget https://github.com/open-mmlab/mmaction2/blob/main/tools/train.py<br />
// wget https://download.openmmlab.com/mmaction/v1.0/skeleton/data/ntu60_2d.pkl #training data at data/skeleton/ntu60_2d.pkl<br />
<br />
!python mmaction2/tools/train.py mmaction2/configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py \<br />
    --work-dir work_dirs/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint \<br />
    --seed 0<br />
<br />
out of memory solved by modify lines (optional):<br />
mmaction2/configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py<br />
	// 84// batch_size=4,<br />
	// 97// batch_size=4,<br />
<br />
run demo_training.ipynb

========================================<br />
Neural Network Architecture Used in demo_training.ipynb:<br />
### HUMAN DETECTION & POSE ESTIMATION<br />
- data/skeleton/ntu60_2d.pkl #stated at mmaction2/configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py<br />
<br />
### ACTION CLASSIFICATION<br />
- mmaction2/configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py<br />
	//checkpoint = --resume, If specify checkpoint path, resume from it, while if not '<br />
        'specify, try to auto resume from the latest checkpoint '<br />
        'in the work directory.<br />
########################################

### custom dataset<br />
- pose_extraction.py #skip ntu_det_postproc step<br />
- place pkl at data/skeleton/<br />
- configure ann_file, pretrained, num_classes inside config file, slowonly_..._.py #<br />

### to do<br />
- mmaction2/demo/visualize_heatmap_volume.py<br />
- where is the .pth file saved after training? ~work_dirs/<br />
- add hash_id to .pth file using publish_model.py<br />
- demo_video_structuralize.py<br />
- training video structuralize<br />
