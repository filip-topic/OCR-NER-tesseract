import os
import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel  
from torch.nn import Sequential
import cv2

# Whitelist YOLO model and any PyTorch classes used in the checkpoint
torch.serialization.add_safe_globals([
    DetectionModel,
    Sequential,
])

INPUT_DIR = './dataset'
OUTPUT_DIR = './predictions'
MODEL_PATH = './model/yolov8n.pt' 

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)

# class ID for Printed Text (PT)
PRINTED_TEXT_CLASS_ID = 2  

# Loop through all images in the input folder
for filename in os.listdir(INPUT_DIR):
    if filename.lower().endswith('.jpg'):
        image_path = os.path.join(INPUT_DIR, filename)
        image = cv2.imread(image_path)
        height, width = image.shape[:2]

        # Inference
        results = model(image)

        for i, r in enumerate(results):
            boxes = r.boxes
            for j, box in enumerate(boxes):
                cls_id = int(box.cls[0])
                if cls_id == PRINTED_TEXT_CLASS_ID:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # Clip to image boundaries
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width, x2), min(height, y2)
                    # Crop and save
                    crop = image[y1:y2, x1:x2]
                    crop_filename = f"{os.path.splitext(filename)[0]}_crop{j}.png"
                    cv2.imwrite(os.path.join(OUTPUT_DIR, crop_filename), crop)

print(f"✅ Printed text crops saved to: {OUTPUT_DIR}")
