Collections:
- Name: HRNet
  Paper:
    Title: Deep high-resolution representation learning for human pose estimation
    URL: http://openaccess.thecvf.com/content_CVPR_2019/html/Sun_Deep_High-Resolution_Representation_Learning_for_Human_Pose_Estimation_CVPR_2019_paper.html
  README: https://github.com/open-mmlab/mmpose/blob/main/docs/src/papers/backbones/hrnet.md

- Config: configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py
  In Collection: HRNet
  Metadata:
    Architecture: &id001
    - HRNet
    Training Data: COCO
  Name: td-hm_hrnet-w32_8xb64-210e_coco-256x192
  Results:
  - Dataset: COCO
    Metrics:
      AP: 0.746
      AP@0.5: 0.904
      AP@0.75: 0.819
      AR: 0.799
      AR@0.5: 0.942
    Task: Body 2D Keypoint
  Weights: https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192-81c58e40_20220909.pth
