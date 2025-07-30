import cv2
import numpy as np
from ultralytics import YOLO
from pyorbbecsdk import *

def crop_center(image, top=0, bottom=0, left=100, right=100):
    h, w = image.shape[:2]
    return image[top:h-bottom, left:w-right]

def get_result_yolo_image(orig_image, model, thickness=2):
    """ Get a result for one single image with your model, drawing 'Copperobject' boxes first. """
    im_size = orig_image.shape
    results = model.predict(orig_image)
    boxes = results[0].boxes  # Bounding boxes
    names = model.names       # Class names

    copper_boxes = []
    other_boxes = []

    for box in boxes:
        class_id = int(box.cls[0])
        class_name = names[class_id]
        xyxy = box.xyxy[0].cpu().numpy().astype(int)  # [x1, y1, x2, y2]

        if class_name == 'Copperobject':
            copper_boxes.append((xyxy, class_name))
        else:
            other_boxes.append((xyxy, class_name))

    # Draw all other objects (red)
    for (xyxy, class_name) in other_boxes:
        x1, y1, x2, y2 = xyxy
        cv2.rectangle(orig_image, (x1, y1), (x2, y2), (0, 0, 255), thickness)
        cv2.putText(orig_image, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

    # Draw 'Copperobject' (green)
    for (xyxy, class_name) in copper_boxes:
        x1, y1, x2, y2 = xyxy
        cv2.rectangle(orig_image, (x1, y1), (x2, y2), (0, 255, 0), thickness)
        cv2.putText(orig_image, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    return orig_image

def show_live_demo(model_path):
    # Load YOLOv8 segmentation model
    model = YOLO(model_path)

    # Initialize Orbbec Femto Bolt camera
    pipeline = Pipeline()
    config = Config()

    # Get color profile (RGB format)
    color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    color_profile = color_profiles.get_default_video_stream_profile()
    config.enable_stream(color_profile)

    # Start streaming
    pipeline.start(config)

    try:
        while True:
            frame_set = pipeline.wait_for_frames(1000)
            color_frame = frame_set.get_color_frame()

            if color_frame is None:
                print("No RGB frame received")
                continue

            # MJPG decoding fix
            rgb_data = color_frame.get_data()
            mjpg_array = np.frombuffer(rgb_data, dtype=np.uint8)
            rgb_image = cv2.imdecode(mjpg_array, cv2.IMREAD_COLOR)  # BGR format

            if rgb_image is None:
                print("Failed to decode MJPG frame")
                continue

            # Optional: convert BGR to RGB if needed by model
            rgb_image_for_model = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            cropped_rgb_image = crop_center(rgb_image, left=250, right=500)
            result_img = get_result_yolo_image(cropped_rgb_image, model)

            # Show
            desired_width = 800
            desired_height = 600
            resized_image = cv2.resize(result_img, (desired_width, desired_height))
            cv2.imshow("Orbbec MJPG - YOLOv8 Segmentation", resized_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

# model_path = '../../model/final_results_from_2_pruning_paper/ef_85.pt'
# model_path = '../../model/all-149-train_val/weights/best.pt'
model_path = '../../model/final_results_from_2_pruning_paper/ef_90.pt'
# model_path = '../../model/all-145-train_val/weights/best.pt'
show_live_demo(model_path)
