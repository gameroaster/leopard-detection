# TENSORFLOW LITE OBJECT DETECTION -  ASSIGNMENT


## OVERVIEW 
This project demonstrates the end-to-end creation of an object detection model optimized for edge devices using YOLOv8 and TensorFlow Lite.
The workflow includes dataset preparation, model training, evaluation, and conversion to .tflite format for deployment on mobile or embedded platforms.

The models detect objects in images and output bounding boxes, class labels, and confidence scores, optimized for edge and mobile deployment using TensorFlow Lite.

### Use Case : PCB Fault Detection

### Objective - Detect PCB manufacturing defects such as:

- Detect leopard presence in images
- Output bounding boxes with confidence scores
- Focus on accuracy and robustness

## DATASET
Source: https://universe.roboflow.com/college-kenac/leopard-detection-88ytd/dataset/2
- 151 manually selected and annotated images

The dataset is a object detection dataset in a ZIP file.
### NOTE : I cant upload the dataset on github since dataset size is exceeding 25mb
The dataset is automatically split into:
- 90% Training
- 10% Validation

### Annotation tool : Label Studio (open source) 
https://labelstud.io/

### Annotation format: Text (txt)

### Classes
-leopard  
  
### The dataset directory structure follows the standard YOLO format:
```bash
  custom_data/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
```

A data.yaml file is programmatically generated in the notebook to define:
- Dataset paths
- Number of classes
- Class names

## Model Architecture

- Model : YOLOv8 Small (yolov8s)
- Framework : Ultralytics YOLO
- Architecture Type : One-stage object detector
- Backbone + Head : CNN-based feature extractor with multi-scale detection heads
- Input Size : 640 × 640

### YOLOv8 is chosen for its:

- High inference speed
- Good accuracy-to-size tradeoff
- Native support for TensorFlow Lite export

## TensorFlow Lite Conversion 
YOLOv8 to TensorFlow Lite Conversion :
This project uses YOLOv8 for object detection and converts the trained model into TensorFlow Lite (TFLite) format for deployment on edge and mobile devices.
Ultralytics provides native support for exporting YOLOv8 models to TFLite, making the conversion process straightforward.

```bash
  pip install ultralytics
  from ultralytics import YOLO
  model = YOLO("yolov8n.pt")
  model.export(format="tflite")
```

## Training Approach

- Pretrained weights : yolov8s.pt
- Training type : Transfer Learning
- Epochs : 60
- Image size : 640
- Optimizer & Loss : Handled internally by Ultralytics YOLO
- Environment : Google Colab with GPU acceleration

Model Evaluation :
- Predictions are generated on validation images.
- Sample detection outputs are visualized directly in the notebook.

This produces a .tflite model suitable for:
- Android application
- Edge devices
- Embedded AI systems

## Output
Each model outputs:

1. Bounding boxes
2. Class labels
3. Confidence scores


![train_batch0](https://github.com/user-attachments/assets/12860b7f-b618-45ae-98ac-5177a2e1fb64)


![train_batch1](https://github.com/user-attachments/assets/8aa975bd-61d8-4fae-a121-dbde70aa45a5)


![train_batch2](https://github.com/user-attachments/assets/aa5ee969-4b02-4214-bb75-c8a95e51b1dc)


### Model files:
- leopard.pt
- yolov8s_float16(2).tflite


## Known Limitations & Challenges

- Quantization not applied
  - The exported TFLite model is not fully integer-quantized, which may limit performance on very low-power devices.

- Dataset size dependency 
  - Model accuracy is highly dependent on dataset size and class balance.

- Hardware constraints
  - Training requires GPU support for reasonable training time.

- Edge deployment tuning
  - Further optimization (INT8 quantization, pruning) may be required for real-time deployment on microcontrollers.

## Future Improvements
- Apply INT8 quantization for better edge performance
- Train larger YOLOv8 variants for higher accuracy
- Perform extensive validation using mAP metrics
- Integrate with Android / Raspberry Pi inference pipelines


## Conclusion

### This project demonstrates an end-to-end object detection pipeline:
- Data preparation
- Model training
- Optimization
- Deployment using TensorFlow Lite

The solution is suitable for embedded and mobile applications.
