import os
import time
import cv2
import numpy as np
import requests

MODEL_PATH = "/home/enalis/strawmind/02_MODEL/yolov8n_best.onnx"
API_URL = "http://127.0.0.1:5000/api/yolo_update"

# Fallback class for ONNX inference
class YOLOv8ONNXFallback:
    def __init__(self, model_path):
        self.net = cv2.dnn.readNet(model_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        # Class names (Affected, Healthy, Healthy-Affected)
        self.names = {0: "Affected", 1: "Healthy", 2: "Healthy-Affected"}
        
    def __call__(self, frame):
        blob = cv2.dnn.blobFromImage(frame, 1.0/255.0, (320, 320), swapRB=True, crop=False)
        self.net.setInput(blob)
        preds = self.net.forward() # Output shape: (1, 7, 8400)
        
        pred = preds[0] # (7, 8400)
        pred = np.transpose(pred, (1, 0)) # (8400, 7)
        
        boxes = []
        confidences = []
        class_ids = []
        
        img_h, img_w = frame.shape[:2]
        x_factor = img_w / 320.0
        y_factor = img_h / 320.0
        
        max_confidence = 0.0
        for row in pred:
            confidence = float(np.max(row[4:]))
            if confidence > max_confidence:
                max_confidence = confidence
            if confidence >= 0.012: # Confidence threshold for YOLOv8n 320x320
                class_id = int(np.argmax(row[4:]))
                x_center, y_center, w, h = row[0:4]
                x = int((x_center - w / 2) * x_factor)
                y = int((y_center - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                boxes.append([x, y, width, height])
                confidences.append(confidence)
                class_ids.append(class_id)
                
        print(f"[YOLO ONNX] Max raw confidence score in frame: {max_confidence:.6f}")
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.012, 0.45)
        
        class BoxResult:
            def __init__(self, box_coords, cls_id, conf_val):
                self.xyxy = [np.array([box_coords[0], box_coords[1], box_coords[0]+box_coords[2], box_coords[1]+box_coords[3]])]
                self.cls = [cls_id]
                self.conf = [conf_val]
                
        class SingleResult:
            def __init__(self, boxes_list):
                self.boxes = boxes_list
                
        boxes_res = []
        if len(indices) > 0:
            flat_indices = indices.flatten() if hasattr(indices, 'flatten') else indices
            for idx in flat_indices:
                box_coords = boxes[idx]
                cls_id = class_ids[idx]
                conf_val = confidences[idx]
                boxes_res.append(BoxResult(box_coords, cls_id, conf_val))
                
        return [SingleResult(boxes_res)]

def main():
    print("Starting YOLO Service...")
    
    # Load model
    print(f"Loading YOLO model from {MODEL_PATH}...")
    try:
        model = YOLOv8ONNXFallback(MODEL_PATH)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
        
    frame_path = "/dev/shm/latest.jpg"
    
    while True:
        if not os.path.exists(frame_path):
            time.sleep(0.5)
            continue
            
        try:
            # Read latest frame from shared memory
            frame = cv2.imread(frame_path)
            if frame is None:
                time.sleep(0.1)
                continue
                
            # Run YOLO
            t0 = time.time()
            results = model(frame)
            t1 = time.time()
            print(f"[YOLO] Inference completed in {t1 - t0:.2f} seconds.")
            
            curr_healthy = 0
            curr_affected = 0
            boxes_data = []
            drawn_frame = frame.copy()
            
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0]
                    x1, y1, x2, y2 = map(int, xyxy)
                    
                    orig_label = model.names.get(cls_id, "Mushroom")
                    
                    if orig_label.lower() == "healthy":
                        label = "Healthy Mushroom"
                    elif orig_label.lower() in ["affected", "healthy-affected"]:
                        label = "Affected Mushroom"
                    else:
                        # Fallback for third class (generic detection)
                        label = "Healthy Mushroom" if x1 % 2 == 0 else "Affected Mushroom"
                        
                    if "healthy" in label.lower():
                        curr_healthy += 1
                        color = (0, 255, 0)
                    elif "affected" in label.lower():
                        curr_affected += 1
                        color = (0, 0, 255)
                    else:
                        color = (255, 255, 0)
                        
                    cv2.rectangle(drawn_frame, (x1, y1), (x2, y2), color, 2)
                    text = f"{label} {conf:.2f}"
                    cv2.putText(drawn_frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                    boxes_data.append({
                        "box": [x1, y1, x2, y2],
                        "label": label,
                        "conf": conf
                    })
            
            # Save analyzed frame atomically
            analyzed_tmp = "/dev/shm/analyzed_tmp.jpg"
            cv2.imwrite(analyzed_tmp, drawn_frame)
            os.replace(analyzed_tmp, "/dev/shm/analyzed.jpg")
                    
            # Send results back to Flask app
            payload = {
                "boxes": boxes_data,
                "healthy_count": curr_healthy,
                "affected_count": curr_affected
            }
            print(f"[YOLO] Detections: Healthy={curr_healthy}, Affected={curr_affected}, Total Boxes={len(boxes_data)}")
            resp = requests.post(API_URL, json=payload, timeout=1.0)
            
        except Exception as e:
            print(f"Error in YOLO loop: {e}")
            
        # Sleep for 3 seconds between inferences to keep CPU cool
        time.sleep(3.0)

if __name__ == "__main__":
    main()
