## Miniconda [quick command line install](https://docs.anaconda.com/free/miniconda/#quick-command-line-install)
Download & install:
```
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
```
Clean up installer:
```
rm -rf ~/miniconda3/miniconda.sh
```
Initialize Miniconda for bash and zsh shells:
```
~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh
```
## Environment Preparation
Create & activate Python 3.8 conda environment:
```
conda create --name action python=3.8 -y
conda activate action
```
[Install PyTorch 2.2.0 + CUDA 12.1](https://pytorch.org/get-started/locally/) and other PyTorch related packages:
```
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```
Check cuda availability:
```python
python  #entering Python terminal:
>>> import torch
>>> torch.cuda.is_available()  #check if cuda enabled and accessible by PyTorch
>>> exit()
```
If false, [install cuda](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_network):
Base Installer (*over network*), **for Ubuntu 22.04**:
```shell
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-5
```
Driver Installer (*legacy kernel*):
```shell
sudo apt-get install -y cuda-drivers #Remember to Enroll MOK during reboot
```
If missed the Enroll MOK screen:
```shell
sudo mokutil -t /var/lib/shim-signed/mok/MOK.der #check if MOK enrolled
sudo mokutil -import /var/lib/shim-signed/mok/MOK.der
sudo mokutil -t /var/lib/shim-signed/mok/MOK.der #verify that MOK enrolled
sudo reboot now #Remember to Enroll MOK, continue, enter password
```
Verify cuda availability:
```python
python  #entering Python terminal:
>>> import torch
>>> torch.cuda.is_available() #verify that cuda enabled and accessible by PyTorch
>>> exit()
```
[Install MMAction2 and all required packages](https://mmaction2.readthedocs.io/en/latest/get_started/installation.html#best-practices):
```shell
pip install -U openmim
mim install mmengine
mim install mmcv
mim install mmdet
mim install mmpose

pip install mmaction2
```
Add drn ([*mmaction2 bug fix*](https://github.com/open-mmlab/mmaction2/issues/2714)):
```shell
pip show mmaction2  # remember the Location
mkdir -p <Location>/mmaction/models/localizers/drn/drn_utils
# Download drn and save to Location
wget -P <Location>/mmaction/models/localizers/drn/drn_utils/ https://raw.githubusercontent.com/open-mmlab/mmaction2/main/mmaction/models/localizers/drn/drn_utils/FPN.py https://raw.githubusercontent.com/open-mmlab/mmaction2/main/mmaction/models/localizers/drn/drn_utils/__init__.py https://raw.githubusercontent.com/open-mmlab/mmaction2/main/mmaction/models/localizers/drn/drn_utils/backbone.py https://raw.githubusercontent.com/open-mmlab/mmaction2/main/mmaction/models/localizers/drn/drn_utils/fcos.py https://raw.githubusercontent.com/open-mmlab/mmaction2/main/mmaction/models/localizers/drn/drn_utils/inference.py https://raw.githubusercontent.com/open-mmlab/mmaction2/main/mmaction/models/localizers/drn/drn_utils/language_module.py https://raw.githubusercontent.com/open-mmlab/mmaction2/main/mmaction/models/localizers/drn/drn_utils/loss.py
wget -P <Location>/mmaction/models/localizers/drn/ https://raw.githubusercontent.com/open-mmlab/mmaction2/main/mmaction/models/localizers/drn/drn.py
```
Install Jupyter Notebook and its supporting packages:
```shell
pip install notebook
pip install ipython
pip install moviepy
```
Install Chrome:
```shell
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
sudo dpkg -i google-chrome-stable_current_amd64.deb
sudo apt-get install -f
```
Install VLC Media Player (*optional*):
```shell
sudo apt install vlc
```
## Skeleton-Based Action Recognition with 3D-CNN
Make directory:
```shell
mkdir -p -v ~/Documents/skripsi/posec3d-video-structuralize
cd ~/Documents/skripsi/posec3d-video-structuralize
```
### Pose Extraction (.pkl creator)
Download **Extraction Script** and save to the proper directory:
```shell
wget https://github.com/open-mmlab/mmaction2/blob/main/tools/data/skeleton/ntu_pose_extraction.py -P mmaction2/tools/data/skeleton/
```
Modify lines:
```shell
gedit mmaction2/tools/data/skeleton/ntu_pose_extraction.py
```
> * Line 15| args.det_config = '../../../demo/demo_configs/faster-rcnn_r50-caffe_fpn_ms-1x_coco-person.py'  # noqa: E501<br />
> * Line 18| args.pose_config = '../../../demo/demo_configs/td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py'  # noqa: E501
Download example video and save to the proper directory:
```shell
wget https://github.com/open-mmlab/mmaction2/blob/main/tools/data/skeleton/S001C001P001R001A001_rgb.avi -P mmaction2/tools/data/skeleton/
```
Download **Pose Detection Config Script** and save to the proper directory:
```shell
wget https://github.com/open-mmlab/mmaction2/blob/main/demo/demo_configs/faster-rcnn_r50-caffe_fpn_ms-1x_coco-person.py -P mmaction2/demo/demo_configs/  # file info https://github.com/open-mmlab/mmdetection/tree/main/configs/faster_rcnn
```
Download **Pose Estimation Config Script** and save to the proper directory:
```shell
wget https://github.com/open-mmlab/mmaction2/blob/main/demo/demo_configs/td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py -P mmaction2/demo/demo_configs/  # file info https://github.com/open-mmlab/mmpose/tree/master/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap
```
Run Extraction Script:
```shell
python ntu_pose_extraction.py S001C001P001R001A001_rgb.avi S001C001P001R001A001_rgb.pkl
```
========================================
<br />Neural Network Architecture Used in ntu_pose_extraction.py:
<br />### HUMAN DETECTION
<br />- config = [mmaction2/demo/demo_configs/faster-rcnn_r50-caffe_fpn_ms-1x_coco-person.py](https://github.com/open-mmlab/mmaction2/blob/main/demo/demo_configs/faster-rcnn_r50-caffe_fpn_ms-1x_coco-person.py)
<br />- checkpoint = [faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth](https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco-person/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth)
<br />### POSE ESTIMATION
<br />- config = [mmaction2/demo/demo_configs/td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py](https://github.com/open-mmlab/mmaction2/blob/main/demo/demo_configs/td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py)
<br />- checkpoint = [hrnet_w32_coco_256x192-c78dce93_20200708.pth](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth)
<br />
<br />[notes](https://github.com/open-mmlab/mmaction2/blob/main/configs/skeleton/posec3d/custom_dataset_training.md):
> One only thing you may need to change is that: since ntu_pose_extraction.py is developed specifically for pose extraction of NTU videos, you can skip the [ntu_det_postproc](https://github.com/aldyraja/posec3d-video-structuralize/blob/master/mmaction2/tools/data/skeleton/ntu_pose_extraction.py#L269) step when using this script for extracting pose from your custom video datasets.

### Testing
Download **NTU60 Label** and save to the proper directory:
```shell
wget https://github.com/open-mmlab/mmaction2/blob/main/configs/_base_/default_runtime.py -P mmaction2/configs/_base_/
wget https://github.com/open-mmlab/mmaction2/blob/main/tools/data/skeleton/label_map_ntu60.txt -P mmaction2/tools/data/skeleton/
```
Download **Pose Detection Config Script** and save to the proper directory:
```shell
wget https://github.com/open-mmlab/mmaction2/blob/main/demo/demo_configs/faster-rcnn_r50_fpn_2x_coco_infer.py -P mmaction2/demo/demo_configs/  # file info https://github.com/open-mmlab/mmdetection/tree/main/configs/faster_rcnn
```
Run demo_testing.ipynb
<br />
<br />========================================
<br />Neural Network Architecture Used in demo_testing.ipynb:
<br />### HUMAN DETECTION
<br />- config = [mmaction2/demo/demo_configs/faster-rcnn_r50_fpn_2x_coco_infer.py](https://github.com/open-mmlab/mmaction2/blob/main/demo/demo_configs/faster-rcnn_r50-caffe_fpn_ms-1x_coco-person.py)
<br />- checkpoint = [faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth](http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth)
<br />### POSE ESTIMATION
<br />- config = [mmaction2/demo/demo_configs/td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py](https://github.com/open-mmlab/mmaction2/blob/main/demo/demo_configs/td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py)
<br />- checkpoint = [hrnet_w32_coco_256x192-c78dce93_20200708.pth](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth)
<br />### ACTION CLASSIFICATION
<br />- config = [mmaction2/configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py](https://github.com/open-mmlab/mmaction2/blob/main/configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py)
<br />- checkpoint = [slowonly_r50_u48_240e_ntu60_xsub_keypoint/slowonly_r50_u48_240e_ntu60_xsub_keypoint-f3adabf1.pth](https://download.openmmlab.com/mmaction/skeleton/posec3d/slowonly_r50_u48_240e_ntu60_xsub_keypoint/slowonly_r50_u48_240e_ntu60_xsub_keypoint-f3adabf1.pth)
<br />
<br />visualize heatmap volume: https://github.com/open-mmlab/mmaction2/blob/4e50a824d3abb708619978de65a30eee2daf81bd/demo/visualize_heatmap_volume.ipynb
<br />########################################

### Training
Download **Train Script** and save to the proper directory:
```
wget https://github.com/open-mmlab/mmaction2/blob/main/tools/train.py -P mmaction2/tools/
```
Download **Pose (.pkl) as Training Data** and save to the proper directory:
```
wget https://download.openmmlab.com/mmaction/v1.0/skeleton/data/ntu60_2d.pkl -P data/skeleton/
```
Run demo_training.ipynb
<br />
<br />========================================
<br />Neural Network Architecture Used in demo_training.ipynb:
<br />### HUMAN DETECTION & POSE ESTIMATION
<br />- [data/skeleton/ntu60_2d.pkl](https://download.openmmlab.com/mmaction/v1.0/skeleton/data/ntu60_2d.pkl)  #stated at [mmaction2/configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py](https://github.com/open-mmlab/mmaction2/blob/main/configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py)
<br />
<br />### ACTION CLASSIFICATION
<br />- config = [mmaction2/configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py](https://github.com/open-mmlab/mmaction2/blob/main/configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py)
<br />- checkpoint = --resume  #If specify checkpoint path, resume from it, while if not specify, try to auto resume from the latest checkpoint in the work directory.
<br />########################################
<br />
<br />### custom dataset
<br />- pose_extraction.py #skip ntu_det_postproc step
<br />- place pkl at data/skeleton/
<br />- configure ann_file, pretrained, num_classes inside config file, slowonly_..._.py #

## Error
### Solve out of memory error lines, Modify lines:
```shell
gedit mmaction2/configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py
```
> * Line 84| batch_size=4,<br />
> * Line 97| batch_size=4,
### Solve imageio_ffmpeg "TypeError: must be real number, not NoneType":
```shell
pip uninstall ffmpeg
pip install moviepy --upgrade --force-reinstall
```
## To Do
- 
