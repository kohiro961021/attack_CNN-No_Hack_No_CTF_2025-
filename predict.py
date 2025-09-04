from ultralytics import YOLO,YOLOv10
import os
import cv2

CLASS_NAMES = ['Green Light', 'Red Light', 'Speed Limit 10', 'Speed Limit 100', 'Speed Limit 110',
               'Speed Limit 120', 'Speed Limit 20', 'Speed Limit 30', 'Speed Limit 40', 'Speed Limit 50',
               'Speed Limit 60', 'Speed Limit 70', 'Speed Limit 80', 'Stop']

def load_models():
    model_v8 = YOLO("models/yolo_v8.pt")
    model_v10 = YOLOv10("models/yolo_v10.pt")
    return model_v8, model_v10
def run_inference(model, image_path, save_path):
    results = model(image_path)[0]

   
    img = cv2.imread(image_path)
    boxes = results.boxes
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            cls_id = int(box.cls)
            label = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"Class {cls_id}"
            conf = float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy.tolist()[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite(save_path, img)

     
        top = boxes[0]
        return {
            "class_id": int(top.cls),
            "class_name": CLASS_NAMES[int(top.cls)] if int(top.cls) < len(CLASS_NAMES) else f"Class {int(top.cls)}",
            "confidence": float(top.conf)
        }
    else:
        cv2.imwrite(save_path, img) 
        return {
            "class_id": -1,
            "class_name": "None",
            "confidence": 0.0
        }
