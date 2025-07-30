# Conveyor Belt Object Detection using YOLO and Orbbec Camera

This project integrates a pre-trained YOLOv11n object detection and instance segmentation model with the Orbbec Femto Bolt depth camera to detect metal scrap objects on a conveyor belt in real time. The goal is to assess the feasibility and performance of using such a system in industrial environments for tasks such as sorting, inspection, and quality control.

## Objective

To evaluate the performance of a pre-trained YOLO object detection model using RGB input from the Orbbec Femto Bolt camera, and to visualize real-time detection through polygon overlays and bounding boxes around objects on a conveyor belt.

## Methodology

1. **Camera Setup**  
   The Orbbec Femto Bolt RGB-D camera is mounted and calibrated to observe the conveyor belt surface.

2. **Data Capture**  
   RGB frames are continuously streamed from the camera as input to the detection pipeline.

3. **Model Inference**  
   A pre-trained YOLOv11n segmentation model is used to detect and segment metal scrap objects in each frame.

4. **Visualization**  
   Detected masks and bounding boxes are rendered using OpenCV to highlight the objects identified by the model.

5. **Performance Evaluation**  
   The system's ability to correctly localize and classify objects is evaluated visually and analytically.

## Dependencies

- Python 3.10
- OpenCV
- NumPy
- PyOrbbecSDK
- Ultralytics YOLO (v11n or compatible)
