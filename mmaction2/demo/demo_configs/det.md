Collections:
  - Name: Faster R-CNN
    Metadata:
      Training Data: COCO
      Training Techniques:
        - SGD with Momentum
        - Weight Decay
      Training Resources: 8x V100 GPUs
      Architecture:
        - FPN
        - RPN
        - ResNet
        - RoIPool
    Paper:
      URL: https://arxiv.org/abs/1506.01497
      Title: "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks"
    README: configs/faster_rcnn/README.md
    Code:
      URL: https://github.com/open-mmlab/mmdetection/blob/v2.0.0/mmdet/models/detectors/faster_rcnn.py#L6
      Version: v2.0.0

  - Name: faster-rcnn_r50_fpn_2x_coco
    In Collection: Faster R-CNN
    Config: configs/faster_rcnn/faster-rcnn_r50_fpn_2x_coco.py
    Metadata:
      Training Memory (GB): 4.0
      inference time (ms/im):
        - value: 46.73
          hardware: V100
          backend: PyTorch
          batch size: 1
          mode: FP32
          resolution: (800, 1333)
      Epochs: 24
    Results:
      - Task: Object Detection
        Dataset: COCO
        Metrics:
          box AP: 38.4
    Weights: https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth
    
  - Name: faster-rcnn_r50-caffe_fpn_ms-2x_coco
    In Collection: Faster R-CNN
    Config: configs/faster_rcnn/faster-rcnn_r50-caffe_fpn_ms-2x_coco.py
    Metadata:
      Training Memory (GB): 4.3
      Epochs: 24
    Results:
      - Task: Object Detection
        Dataset: COCO
        Metrics:
          box AP: 39.7
    Weights: https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_2x_coco/faster_rcnn_r50_caffe_fpn_mstrain_2x_coco_bbox_mAP-0.397_20200504_231813-10b2de58.pth

  - Name: faster-rcnn_r50-caffe_fpn_ms-3x_coco
    In Collection: Faster R-CNN
    Config: configs/faster_rcnn/faster-rcnn_r50-caffe_fpn_ms-3x_coco.py
    Metadata:
      Training Memory (GB): 3.7
      Epochs: 36
    Results:
      - Task: Object Detection
        Dataset: COCO
        Metrics:
          box AP: 39.9
    Weights: https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco_20210526_095054-1f77628b.pth
