<a href="https://colab.research.google.com/github/jhgan00/cxr-detector/blob/master/cxr_detector.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# CXR Detector

> CXR 사진에서 L 문자 디텍션
>
> - 모델:  [YOLO v3](https://github.com/AlexeyAB/darknet)
>
> - mAP\@0.5 on the test dataset: 0.9807 
>
> - mAP\@0.5 on the [RSNA pneumonia detection challenge dataset](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge): 0.1557 

# 0. Build

```
import os
os.chdir("./drive/My Drive/")
!git clone https://github.com/AlexeyAB/darknet.git
os.chdir("darknet")
!sed '1 s/^.*$/GPU=1/; 2 s/^.*$/CUDNN=1/' -i Makefile
!make
```

# 1. Train



## 1.1 Config

- \# of train images: 1,800
- \# of valid images: 200
- \# of test images: 200

params|values
---|---
batch|64
subdivisions|16
width|416
height|416
channels|3
momentum|0.9
decay|0.0005
angle|0
saturation|1.5
exposure|1.5
hue|0.1
learning_rate|0.001
burn_in|1000
max_batches|6000
policy|steps
steps|4800,5400
scales|0.1,0.1

## 1.2. Training



```python
import os
os.chdir("drive/My Drive/darknet")
!chmod +x darknet
```


```python
!cat cfg/cxr.data
```

- Start training with Darknet53.Conv.74 weights pretrained for classification on the ImageNet dataset 


```python
!./darknet detector train \
    cfg/cxr.data \
    cfg/cxr_yolov3_train.cfg \
    darknet53.conv.74 \
    -dont_show -map
```

- Restore the latest checkpoint & continue training


```python
!./darknet detector train \
    cfg/cxr.data \
    cfg/cxr_yolov3_train.cfg \
    backup/cxr_yolov3_train_last.weights \
    -dont_show -map
```

## 1.3. Check validation samples

#### 1.3.1. Output images


```python
!./darknet detector test \
    cfg/cxr.data \
    cfg/cxr_yolov3_train.cfg \
    backup/cxr_yolov3_train_best.weights \
    images/000-pgan-cxr_abnormal-preset-v2-1gpu-fp32-network-snapshot-014000-001285.png
```


```python
import matplotlib.pyplot as plt
plt.figure(figsize = (10,10))
plt.imshow(plt.imread("predictions.jpg"))
```




    <matplotlib.image.AxesImage at 0x7fae00285828>


<img src="./assets/output_17_1.png" width="350">


### 1.3.2. **1.00 mAP** at IoU=50




```bash
detections_count = 282, unique_truth_count = 200  
class_id = 0, name = L, ap = 100.00%     (TP = 200, FP = 0) 

 for conf_thresh = 0.25, precision = 1.00, recall = 1.00, F1-score = 1.00 
 for conf_thresh = 0.25, TP = 200, FP = 0, FN = 0, average IoU = 77.03 % 

 IoU threshold = 50 %, used Area-Under-Curve for each unique Recall 
 mean average precision (mAP@0.50) = 1.000000, or 100.00 % 
Total Detection Time: 145 Seconds
```


```python
!./darknet detector map \
    cfg/cxr.data \
    cfg/cxr_yolov3_train.cfg \
    backup/cxr_yolov3_train_best.weights
```

# 2. Test

- mAP after 1,600 iterations(batches)

## 2.1. PGAN Test dataset: **0.9942 mAP** at IoU=50

```
 detections_count = 274, unique_truth_count = 197  
class_id = 0, name = L, ap = 99.42%      (TP = 194, FP = 0) 

 for conf_thresh = 0.25, precision = 1.00, recall = 0.98, F1-score = 0.99 
 for conf_thresh = 0.25, TP = 194, FP = 0, FN = 3, average IoU = 79.09 % 

 IoU threshold = 50 %, used Area-Under-Curve for each unique Recall 
 mean average precision (mAP@0.50) = 0.994191, or 99.42 % 
```


```python
!./darknet detector map \
    cfg/cxr-test.data \
    cfg/cxr_yolov3_test.cfg \
    backup/cxr_yolov3_train_best.weights
```

## 2.2. RSNA Pneumonia detection dataset: 0.1232 mAP at IoU=50

<img src="./assets/rsna3.png" width="350">

```
detections_count = 1049, unique_truth_count = 181  
class_id = 0, name = L, ap = 12.32%      (TP = 43, FP = 143) 

 for conf_thresh = 0.25, precision = 0.23, recall = 0.24, F1-score = 0.23 
 for conf_thresh = 0.25, TP = 43, FP = 143, FN = 138, average IoU = 14.08 % 

 IoU threshold = 50 %, used Area-Under-Curve for each unique Recall 
 mean average precision (mAP@0.50) = 0.123228, or 12.32 % 
```


```python
!./darknet detector map \
    cfg/cxr-kaggle-test.data \
    cfg/cxr_yolov3_test.cfg \
    backup/cxr_yolov3_train_best.weights
```

# 3. Improving detection performance by data augmentation

- [CFG Parameters in the `[net]` section](https://github.com/AlexeyAB/darknet/wiki/CFG-Parameters-in-the-%5Bnet%5D-section)

## 3.1. Config

- Add blur, mosaic, gaussian noise


params|values
---|---
batch|64
subdivisions|16
width|416
height|416
channels|3
momentum|0.9
decay|0.0005
angle|0
saturation|1.5
exposure|1.5
hue|0.1
learning_rate|0.001
burn_in|1000
max_batches|6000
policy|steps
steps|4800,5400
scales|0.1,0.1
blur|1
mosaic|1
gaussian_noise|1

## 3.2. Training

- Start training with Darknet53.Conv.74 weights pretrained for classification on the ImageNet dataset 


```python
!./darknet detector train \
    cfg/cxr-aug.data \
    cfg/cxr_yolov3_train_aug.cfg \
    darknet53.conv.74 \
    -dont_show -map
```

- Restore the latest checkpoint & continue training


```python
!./darknet detector train \
    cfg/cxr-aug.data \
    cfg/cxr_yolov3_train_aug.cfg \
    aug-backup/cxr_yolov3_train_aug_last.weights \
    -dont_show -map
```

## 3.3. Validation & Test

- mAP after 1,600 iterations(batches)

### 3.3.1. Validation: 1.00 mAP at IoU=50


```python
!./darknet detector map \
    cfg/cxr.data \
    cfg/cxr_yolov3_test.cfg \
    aug-backup/cxr_yolov3_train_aug_best.weights \
```

```
class_id = 0, name = L, ap = 100.00%     (TP = 200, FP = 0) 

for conf_thresh = 0.25, precision = 1.00, recall = 1.00, F1-score = 1.00 
for conf_thresh = 0.25, TP = 200, FP = 0, FN = 0, average IoU = 77.97 % 

IoU threshold = 50 %, used Area-Under-Curve for each unique Recall 
 mean average precision (mAP@0.50) = 1.000000, or 100.00 % 
```

### 3.3.2. PGAN test dataset: 0.9807 mAP at IoU=50

- We have slight decrease of mAP on the test dataset 


```python
!./darknet detector map \
    cfg/cxr-test.data \
    cfg/cxr_yolov3_test.cfg \
    aug-backup/cxr_yolov3_train_aug_best.weights \
```

```
class_id = 0, name = L, ap = 98.07%      (TP = 193, FP = 3) 

for conf_thresh = 0.25, precision = 0.98, recall = 0.98, F1-score = 0.98 
for conf_thresh = 0.25, TP = 193, FP = 3, FN = 4, average IoU = 70.01 % 

IoU threshold = 50 %, used Area-Under-Curve for each unique Recall 
mean average precision (mAP@0.50) = 0.980706, or 98.07 % 
Total Detection Time: 6 Seconds
```

### 3.3.3. RSNA pneumonia detection dataset: 0.1557 mAP at IoU=50

- We have slight increase of mAP on the RSNA pneumonia detection dataset 



```python
!./darknet detector map \
    cfg/cxr-kaggle-test.data \
    cfg/cxr_yolov3_test.cfg \
    aug-backup/cxr_yolov3_train_aug_best.weights \
```

```
class_id = 0, name = L, ap = 15.57%      (TP = 52, FP = 202) 

for conf_thresh = 0.25, precision = 0.20, recall = 0.29, F1-score = 0.24 
for conf_thresh = 0.25, TP = 52, FP = 202, FN = 129, average IoU = 12.61 % 

IoU threshold = 50 %, used Area-Under-Curve for each unique Recall 
mean average precision (mAP@0.50) = 0.155661, or 15.57 % 
Total Detection Time: 5 Seconds
```

# 참고자료

- [darknet](https://github.com/AlexeyAB/darknet)
- [RSNA pneumonia detection challenge dataset](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge)
- [Yolo_Label](https://github.com/developer0hye/Yolo_Label)
