### Declaration
本代码库仅对几个代码库做了整理，串了下流程，代码库分别为：
- 人脸检测：[yolov5-face](https://github.com/Rianusr/yolov5-face.git)
- 人脸对齐：[insightface](https://github.com/deepinsight/insightface/blob/master/python-package/insightface/utils/face_align.py)
- 人脸识别：[Elasticface](https://github.com/hoangviet661999/Elasticface.git)

本代码库所用到的模型，都可以从以上代码库直接下载预训练模型。

### 代码目录结构
```
    .
    ├── ckpts
    │   ├── arc+295672backbone.pth          ## elasticFace pretrainedModel
    │   └── yolov5m-face.pt                 ## yolo5-face pretrainedModel
    ├── data
    │   ├── aligned_results                 ## 对齐人脸保存位置
    │   ├── face_db                         ## 整理好的人脸数据库， 存放仅包含单个人脸的图片
    │   ├── images                          ## 需要识别的图片地址
    │   └── recog_results                   ## 人脸识别可视化结果
    ├── face_align.py                       ## 人脸对齐
    ├── face_detector.py                    ## 人脸检测
    ├── face_encoding.py                    ## 人脸特征编码
    ├── face_recognition.py                 ## 人脸识别
    ├── models
    │   ├── common.py
    │   ├── elasticFace
    │   ├── experimental.py
    │   ├── __init__.py
    │   └── yolo.py
    ├── README.md
    └── utils
        ├── countFLOPS.py
        ├── face_utils.py
        ├── general.py
        ├── __init__.py
        └── torch_utils.py
```

### 快速开始
```
git clone 
cd yolov5ElasticFaceRecognize
python face_recognition.py
```