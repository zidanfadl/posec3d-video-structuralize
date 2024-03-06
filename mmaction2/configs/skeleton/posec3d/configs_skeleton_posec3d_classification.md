# Collections:
  - Name: PoseC3D
    README: configs/skeleton/posec3d/README.md
    Paper:
      URL: https://arxiv.org/abs/2104.13586
      Title: "Revisiting Skeleton-based Action Recognition"

# Models:
## config_recognition
### demo: slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py
  - Name: slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint
    Config: configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py
    In Collection: PoseC3D
    Metadata:
      Architecture: SlowOnly-R50
      Batch Size: 16
      Epochs: 240
      FLOPs: 20.6G
      Parameters: 2.0M
      Training Data: NTU60-XSub
      Training Resources: 8 GPUs
      pseudo heatmap: keypoint
    Results:
    - Dataset: NTU60-XSub
      Task: Skeleton-based Action Recognition
      Metrics:
        Top 1 Accuracy: 93.6
    Training Log: https://download.openmmlab.com/mmaction/v1.0/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.log
    Weights: https://download.openmmlab.com/mmaction/v1.0/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint_20220815-38db104b.pth

### 2019: slowonly_r50_u48_240e_ntu120_xsub_keypoint.py


## checkpoint_recognition
demo: https://download.openmmlab.com/mmaction/skeleton/posec3d/slowonly_r50_u48_240e_ntu60_xsub_keypoint/slowonly_r50_u48_240e_ntu60_xsub_keypoint-f3adabf1.pth
2019: work_dirs/slowonly_r50_u48_240e_ntu120_xsub_keypoint/best_top1_acc_epoch_90_8.pth
