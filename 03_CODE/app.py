import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import time
import threading
import cv2
import numpy as np
import requests
from flask import Flask, render_template, Response, jsonify, request, send_file
# Thử import ultralytics. Nếu không có (như trên Raspberry Pi 3 32-bit), định nghĩa bộ đọc ONNX bằng OpenCV DNN
try:
    from ultralytics import YOLO
    has_ultralytics = True
except ImportError:
    has_ultralytics = False
    print("Không tìm thấy ultralytics. Hệ thống sẽ sử dụng OpenCV DNN làm giải pháp dự phòng chạy mô hình ONNX.")

class YOLOv8ONNXFallback:
    def __init__(self, model_path):
        self.net = cv2.dnn.readNet(model_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.names = {0: "Healthy", 1: "Affected"}
        
    def __call__(self, frame):
        blob = cv2.dnn.blobFromImage(frame, 1.0/255.0, (640, 640), swapRB=True, crop=False)
        self.net.setInput(blob)
        preds = self.net.forward() # Output shape: (1, 6, 8400)
        
        pred = preds[0] # (6, 8400)
        pred = np.transpose(pred, (1, 0)) # (8400, 6)
        
        boxes = []
        confidences = []
        class_ids = []
        
        img_h, img_w = frame.shape[:2]
        x_factor = img_w / 640.0
        y_factor = img_h / 640.0
        
        max_confidence_found = 0.0
        for row in pred:
            confidence = float(np.max(row[4:]))
            if confidence > max_confidence_found:
                max_confidence_found = confidence
            if confidence >= 0.35:
                class_id = int(np.argmax(row[4:]))
                x_center, y_center, w, h = row[0:4]
                x = int((x_center - w / 2) * x_factor)
                y = int((y_center - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                boxes.append([x, y, width, height])
                confidences.append(confidence)
                class_ids.append(class_id)
                
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.35, 0.45)
        
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
            # OpenCV DNN NMSBoxes returns a flat array of indices in OpenCV 4.x
            flat_indices = indices.flatten() if hasattr(indices, 'flatten') else indices
            for idx in flat_indices:
                box_coords = boxes[idx]
                cls_id = class_ids[idx]
                conf_val = confidences[idx]
                boxes_res.append(BoxResult(box_coords, cls_id, conf_val))
                
        # print(f"[YOLO ONNX] Max raw confidence score in image: {max_confidence_found:.4f}")
        return [SingleResult(boxes_res)]

app = Flask(__name__)

# Cấu hình lưu trữ log bệnh nấm rơm
import json
import shutil
from datetime import datetime

DISEASE_SNAPSHOTS_DIR = os.path.join(os.path.dirname(__file__), "../04_DATA/disease_snapshots")
DISEASE_LOGS_FILE = os.path.join(os.path.dirname(__file__), "../04_DATA/disease_logs.json")

os.makedirs(DISEASE_SNAPSHOTS_DIR, exist_ok=True)
if not os.path.exists(DISEASE_LOGS_FILE):
    with open(DISEASE_LOGS_FILE, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False)

last_disease_log_time = 0
COOLDOWN_SECONDS = 120 # 2 phút chờ tránh ghi trùng lặp liên tục

# Khóa luồng để bảo vệ tệp tin ghi log
log_file_lock = threading.Lock()

def save_disease_log_async(affected_count, temp_val, hum_val, current_time):
    try:
        # 1. Lưu ảnh chụp phân tích từ shared memory sang thư mục lưu trữ vĩnh viễn trên SD card
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_filename = f"snapshot_{timestamp_str}.jpg"
        local_snapshot_path = os.path.join(DISEASE_SNAPSHOTS_DIR, snapshot_filename)
        
        analyzed_source = "/dev/shm/analyzed.jpg"
        if os.path.exists(analyzed_source):
            try:
                shutil.copy(analyzed_source, local_snapshot_path)
            except Exception as e:
                print(f"[Async Log] Lỗi khi sao chép ảnh chụp bệnh: {e}")
                
        # 2. Tạo khuyến nghị điều trị tự động dựa trên thông số lúc phát hiện
        recs = [
            "Khẩn cấp cách ly và tiêu hủy tai nấm nhiễm bệnh khỏi khay trồng để tránh bào tử phát tán.",
        ]
        if temp_val > 32.0:
            recs.append(f"Nhiệt độ hiện tại rất cao ({temp_val}°C). Hãy bật quạt tản nhiệt để hạ xuống dưới 30°C giúp nấm phát triển khỏe mạnh.")
        elif temp_val < 28.0:
            recs.append(f"Nhiệt độ hiện tại thấp ({temp_val}°C). Hãy che chắn luống nấm hoặc tắt quạt tản nhiệt để giữ nhiệt độ ấm áp (28°C - 32°C).")
            
        if hum_val < 85.0:
            recs.append(f"Độ ẩm hiện tại hơi thấp ({hum_val}%). Hãy bật máy phun sương để tăng độ ẩm đạt mức tối ưu 85% - 90%.")
        elif hum_val > 92.0:
            recs.append(f"Độ ẩm hiện tại quá cao ({hum_val}%). Hãy tạm thời tắt phun sương và tăng cường thông gió tránh thối chân nấm.")
            
        new_log = {
            "id": f"log_{int(current_time)}",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "disease_type": "Affected Mushroom (Nấm nhiễm bệnh)",
            "count": affected_count,
            "temp": temp_val,
            "hum": hum_val,
            "image_url": f"/disease_snapshot/{snapshot_filename}",
            "recommendations": recs
        }
        
        # 3. Đọc và ghi log an toàn bằng cách khóa tài nguyên file
        with log_file_lock:
            logs = []
            if os.path.exists(DISEASE_LOGS_FILE):
                try:
                    with open(DISEASE_LOGS_FILE, "r", encoding="utf-8") as f:
                        logs = json.load(f)
                except Exception as e:
                    print(f"[Async Log] Lỗi đọc file log: {e}. Khởi tạo danh sách mới.")
                    logs = []
                    
            logs.insert(0, new_log)
            logs = logs[:50] # Giới hạn 50 dòng log gần nhất
            
            with open(DISEASE_LOGS_FILE, "w", encoding="utf-8") as f:
                json.dump(logs, f, ensure_ascii=False, indent=2)
            print(f"[Async Log] Đã ghi nhận bệnh nấm rơm thành công: {new_log['id']}")
            
    except Exception as e:
        print(f"[Async Log] Lỗi nghiêm trọng trong luồng ghi log ngầm: {e}")



# Cấu hình mặc định (sử dụng Camera cục bộ và IP cảm biến thực tế của bạn)
CAMERA_INDEX = 0
ESP32_SENSOR_IP = "10.90.118.237"

# Biến lưu trữ dữ liệu cảm biến và số liệu thống kê AI
sensor_data = {
    "temp": 28.5,
    "hum": 82.0,
    "eco2": 450,
    "aqi": 1,
    "fan": 0,
    "fogger": 0,
    "mode": "auto",
    "status": "Chưa kết nối ESP32-WROOM",
    "healthy_count": 0,
    "affected_count": 0,
    "disease_detected": False,
    "analysis_time": 0
}

# Khóa để đồng bộ dữ liệu cảm biến và luồng camera
data_lock = threading.Lock()
frame_lock = threading.Lock()

# Biến kiểm tra và khung hình được AI xử lý mới nhất để stream
is_custom_model = False
last_cam_retry_time = 0
latest_processed_frame = None

# Xác định xem có dùng model ONNX hay không
use_onnx = not has_ultralytics

# Đường dẫn đến file model YOLO
MODEL_PATH = None

# Nếu bắt buộc dùng ONNX (do không có ultralytics)
if use_onnx:
    for path in ["yolov8s_best.onnx", "best.onnx", "../02_MODEL/yolov8s_best.onnx", "02_MODEL/yolov8s_best.onnx"]:
        if os.path.exists(path):
            MODEL_PATH = path
            break
else:
    for path in ["yolov8s_best.pt", "best.pt", "../02_MODEL/yolov8s_best.pt", "02_MODEL/yolov8s_best.pt", "yolov8s_best.onnx", "best.onnx", "../02_MODEL/yolov8s_best.onnx"]:
        if os.path.exists(path):
            MODEL_PATH = path
            break

if MODEL_PATH is None:
    # Fallback mặc định
    MODEL_PATH = "yolov8s.pt"

model = None
is_custom_model = True
print("Sẽ tải mô hình YOLOv8s ONNX trong luồng nhận diện ngầm.")

# Biến cờ giả lập khi không kết nối được thiết bị vật lý
use_cam_simulation = False
use_sensor_simulation = False

# Luồng ngầm: Liên tục cập nhật dữ liệu cảm biến và kiểm tra kịch bản điều khiển tự động
def sensor_polling_thread():
    global sensor_data, use_sensor_simulation
    print("Bắt đầu luồng đọc dữ liệu cảm biến...")
    
    # Biến lưu trạng thái cũ để tránh gửi API trùng lặp
    last_fan_state = -1
    last_fogger_state = -1
    
    while True:
        try:
            url = f"http://{ESP32_SENSOR_IP}/data"
            response = requests.get(url, timeout=2.0)
            
            if response.status_code == 200:
                data = response.json()
                with data_lock:
                    sensor_data.update(data)
                    sensor_data["status"] = "Đang kết nối"
                use_sensor_simulation = False
            else:
                raise Exception("Lỗi HTTP Code")
                
        except Exception as e:
            # Ghi nhận trạng thái mất kết nối và chuyển sang giả lập
            with data_lock:
                sensor_data["status"] = "Mất kết nối (Giả lập)"
            use_sensor_simulation = True
            
        # Nếu ở chế độ Giả lập (không kết nối được board thật)
        if use_sensor_simulation:
            with data_lock:
                # Tạo dữ liệu giả lập dao động quanh ngưỡng kiểm thử
                # Nhiệt độ dao động quanh 30-33°C, Độ ẩm dao động quanh 83-87%
                t_wave = 32.0 + 1.5 * np.sin(time.time() / 15.0)
                h_wave = 85.0 - 4.0 * np.cos(time.time() / 20.0)
                
                sensor_data["temp"] = round(t_wave, 1)
                sensor_data["hum"] = round(h_wave, 1)
                sensor_data["eco2"] = int(450 + 50 * np.sin(time.time() / 10.0))
                sensor_data["aqi"] = 1 if sensor_data["eco2"] < 600 else 2
                
                # Thực hiện logic điều khiển giả lập trực tiếp trên cache
                if sensor_data["mode"] == "auto":
                    if sensor_data["temp"] > 32.0:
                        sensor_data["fan"] = 1
                    else:
                        sensor_data["fan"] = 0
                        
                    if sensor_data["hum"] < 85.0:
                        sensor_data["fogger"] = 1
                    else:
                        sensor_data["fogger"] = 0

        # Nếu đang kết nối thật, ESP32 sẽ tự đảm nhận việc điều khiển tự động.
        # Luồng PC chỉ cần polling lấy dữ liệu cập nhật trạng thái hiển thị.
        elif not use_sensor_simulation:
            pass

        time.sleep(2.0)



# Tạo luồng video giả lập (sử dụng khi mất kết nối ESP32-CAM hoặc để demo)
def generate_mock_frame():
    # Tạo một khung hình nền đen/xám với hiệu ứng động
    width, height = 640, 480
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Vẽ nền trang trại nấm rơm giả lập
    cv2.rectangle(frame, (0, 0), (width, height), (30, 25, 20), -1)
    
    # Vẽ một vài chiếc nấm rơm (hình tròn/oval) làm mẫu giả lập
    # Nấm 1: Khỏe mạnh (Healthy)
    cv2.ellipse(frame, (200, 240), (40, 50), 0, 0, 360, (200, 200, 200), -1)
    cv2.rectangle(frame, (185, 280), (215, 330), (160, 160, 160), -1)
    
    # Nấm 2: Bị bệnh (Affected)
    cv2.ellipse(frame, (420, 220), (35, 45), 0, 0, 360, (140, 150, 140), -1)
    # Vẽ vết đốm bệnh màu nâu vàng trên nấm 2
    cv2.circle(frame, (410, 210), 8, (50, 80, 120), -1)
    cv2.circle(frame, (430, 230), 6, (50, 80, 120), -1)
    cv2.rectangle(frame, (405, 260), (435, 310), (130, 130, 130), -1)
    
    # Vẽ luống rơm nền
    cv2.line(frame, (0, 320), (width, 320), (40, 100, 150), 4)
    for i in range(0, width, 30):
        cv2.line(frame, (i, 320), (i + 15, 360), (30, 80, 120), 2)
        
    return frame

# Hàm chạy một tác vụ với timeout để tránh treo luồng khi phần cứng camera gặp lỗi (như lỗi CSI No data received)
def run_with_timeout(func, args=(), timeout=2.5):
    result = [None, None]
    def target():
        try:
            result[0] = func(*args)
        except Exception as e:
            result[1] = e
            
    t = threading.Thread(target=target)
    t.daemon = True
    t.start()
    t.join(timeout)
    if t.is_alive():
        return None, TimeoutError("Thao tác bị treo quá thời gian cho phép")
    return result[0], result[1]

def open_camera_fn(index):
    c = cv2.VideoCapture(index)
    if c.isOpened():
        c.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        c.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return c
    return None

# Luồng ngầm: Chụp ảnh từ camera cục bộ của Pi 3 và xử lý nhận diện AI (chạy 1 lần cho mọi tab client)
latest_raw_frame = None
raw_frame_lock = threading.Lock()
last_boxes = []
boxes_lock = threading.Lock()

# Luồng camera chỉ làm nhiệm vụ lấy ảnh và vẽ bounding box, chạy siêu mượt 15 FPS
latest_raw_frame = None
raw_frame_lock = threading.Lock()
last_boxes = []
boxes_lock = threading.Lock()

# Luồng camera chỉ làm nhiệm vụ lấy ảnh và vẽ bounding box, chạy siêu mượt 15 FPS
def camera_ai_thread():
    global sensor_data, use_cam_simulation, last_cam_retry_time, latest_processed_frame, is_custom_model, CAMERA_INDEX, last_boxes
    print("Bắt đầu luồng đọc camera cục bộ...")
    
    cap = None
    camera_hardware_failed = False
    frame_path = "/dev/shm/latest.jpg"
    
    while True:
        frame = None
        current_time = time.time()
        
        # Tự động kết nối lại nếu bị lỗi phần cứng (thử lại mỗi 30 giây để khôi phục tự động)
        if camera_hardware_failed:
            use_cam_simulation = True
            if current_time - last_cam_retry_time > 30.0:
                last_cam_retry_time = current_time
                print("[Camera Pi] Đang thử kết nối lại camera sau khi lỗi phần cứng...")
                res, err = run_with_timeout(open_camera_fn, (CAMERA_INDEX,), timeout=3.5)
                if err is None and res is not None:
                    print("[Camera Pi] Đã khôi phục kết nối thành công với Camera thật!")
                    cap = res
                    use_cam_simulation = False
                    camera_hardware_failed = False
            
        # Nếu chưa mở camera thật, tiến hành mở thiết bị
        if not use_cam_simulation and not camera_hardware_failed:
            if cap is None:
                print(f"[Camera Pi] Đang mở camera tại index {CAMERA_INDEX}...")
                res, err = run_with_timeout(open_camera_fn, (CAMERA_INDEX,), timeout=3.5)
                if err is None and res is not None:
                    cap = res
                    print(f"[Camera Pi] Đã mở thành công Camera cục bộ tại index {CAMERA_INDEX}")
                    use_cam_simulation = False
                else:
                    print(f"[Camera Pi] Mở camera tại index {CAMERA_INDEX} thất bại hoặc treo. Lỗi: {err}. Chuyển hoàn toàn sang giả lập.")
                    use_cam_simulation = True
                    camera_hardware_failed = True
                    last_cam_retry_time = current_time
                    
        # Cơ chế TỰ ĐỘNG KẾT NỐI LẠI (Chỉ chạy khi không có lỗi phần cứng nghiêm trọng)
        elif not camera_hardware_failed:
            if current_time - last_cam_retry_time > 10.0:
                last_cam_retry_time = current_time
                print(f"[Camera Pi] Đang thử kết nối lại camera tại index {CAMERA_INDEX}...")
                res, err = run_with_timeout(open_camera_fn, (CAMERA_INDEX,), timeout=3.5)
                if err is None and res is not None:
                    print("[Camera Pi] Đã tự động kết nối lại thành công với Camera thật!")
                    cap = res
                    use_cam_simulation = False
                else:
                    if res is not None:
                        try:
                            res.release()
                        except Exception:
                            pass
                    
        # Đọc khung hình từ Camera thật
        if not use_cam_simulation and cap is not None and not camera_hardware_failed:
            # Chạy cap.read() với timeout để tránh treo khi cảm biến không gửi dữ liệu
            res, err = run_with_timeout(cap.read, (), timeout=1.5)
            ret, raw_frame = res if (err is None and res is not None) else (False, None)
            
            if ret and raw_frame is not None:
                frame = raw_frame
                tmp_path = "/dev/shm/latest_tmp.jpg"
                cv2.imwrite(tmp_path, frame)
                os.replace(tmp_path, frame_path)
            else:
                print(f"[Camera Pi] Không thể đọc khung hình từ Camera hoặc thao tác bị treo. Lỗi: {err}. Chuyển hoàn toàn sang giả lập.")
                try:
                    cap.release()
                except Exception:
                    pass
                cap = None
                use_cam_simulation = True
                camera_hardware_failed = True
                last_cam_retry_time = current_time
                
        # Nếu đang ở chế độ giả lập ảnh (simulation)
        if use_cam_simulation or frame is None:
            frame = generate_mock_frame()
            
            cv2.putText(frame, "SIMULATION MODE (NO CAMERA)", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            cycle_time = int(time.time()) % 15
            if cycle_time < 7:
                demo_labels = [
                    {"box": [80, 160, 200, 300], "label": "Healthy Mushroom", "conf": 0.95},
                    {"box": [220, 180, 320, 320], "label": "Healthy Mushroom", "conf": 0.91},
                    {"box": [360, 200, 470, 340], "label": "Healthy Mushroom", "conf": 0.88}
                ]
            else:
                demo_labels = [
                    {"box": [80, 160, 200, 300], "label": "Healthy Mushroom", "conf": 0.95},
                    {"box": [220, 180, 320, 320], "label": "Healthy Mushroom", "conf": 0.91},
                    {"box": [360, 160, 480, 320], "label": "Affected Mushroom", "conf": 0.88},
                    {"box": [480, 200, 580, 340], "label": "Affected Mushroom", "conf": 0.82}
                ]
            
            healthy_count = sum(1 for item in demo_labels if "Healthy" in item["label"])
            affected_count = sum(1 for item in demo_labels if "Affected" in item["label"])
            
            for item in demo_labels:
                x1, y1, x2, y2 = item["box"]
                lbl = item["label"]
                conf = item["conf"]
                color = (0, 200, 0) if "Healthy" in lbl else (0, 0, 220)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                text = f"{lbl} {conf:.2f}"
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            
            with data_lock:
                sensor_data["healthy_count"] = healthy_count
                sensor_data["affected_count"] = affected_count
                sensor_data["disease_detected"] = affected_count > 0
        else:
            pass
                
        # Cập nhật khung hình đã xử lý vào cache
        with frame_lock:
            latest_processed_frame = frame.copy()
            
        time.sleep(0.06)

def video_stream_generator():
    global latest_processed_frame
    
    while True:
        frame = None
        with frame_lock:
            if latest_processed_frame is not None:
                frame = latest_processed_frame.copy()
                
        if frame is None:
            # Khung hình chờ tạm thời nếu luồng camera chưa khởi động xong
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Waiting for Pi Camera...", (50, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            time.sleep(0.05)
            continue
            
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
               
        time.sleep(0.06)

# Khởi chạy luồng cảm biến
polling_thread = threading.Thread(target=sensor_polling_thread, daemon=True)
polling_thread.start()

# Khởi chạy luồng camera & AI chạy ngầm để chia sẻ tài nguyên cho nhiều tab xem cùng lúc
cam_thread = threading.Thread(target=camera_ai_thread, daemon=True)
cam_thread.start()









# Router chính
# API tiếp nhận cập nhật kết quả nhận diện từ tiến trình YOLO độc lập
@app.route('/api/yolo_update', methods=['POST'])
def yolo_update():
    global last_boxes, sensor_data, last_disease_log_time
    req_data = request.json
    
    new_boxes = req_data.get("boxes", [])
    healthy = req_data.get("healthy_count", 0)
    affected = req_data.get("affected_count", 0)
    
    with boxes_lock:
        last_boxes = new_boxes
        
    with data_lock:
        sensor_data["healthy_count"] = healthy
        sensor_data["affected_count"] = affected
        sensor_data["disease_detected"] = affected > 0
        sensor_data["analysis_time"] = time.time()
        
    # Logic ghi log khi phát hiện bệnh nấm rơm
    if affected > 0:
        current_time = time.time()
        if current_time - last_disease_log_time >= COOLDOWN_SECONDS:
            last_disease_log_time = current_time
            
            # Đọc cảm biến thực tế để đưa vào gợi ý điều trị
            temp_val = sensor_data.get("temp", 28.5)
            hum_val = sensor_data.get("hum", 82.0)
            
            # Khởi chạy luồng ghi log ngầm bất đồng bộ
            log_thread = threading.Thread(
                target=save_disease_log_async,
                args=(affected, temp_val, hum_val, current_time),
                daemon=True
            )
            log_thread.start()
                
    return jsonify({"status": "success"})

@app.route('/api/disease_logs', methods=['GET'])
def get_disease_logs():
    try:
        if os.path.exists(DISEASE_LOGS_FILE):
            with open(DISEASE_LOGS_FILE, "r", encoding="utf-8") as f:
                logs = json.load(f)
            return jsonify(logs)
    except Exception as e:
        print(f"Lỗi khi đọc file logs: {e}")
    return jsonify([])

@app.route('/disease_snapshot/<filename>')
def get_disease_snapshot(filename):
    from flask import send_from_directory
    return send_from_directory(DISEASE_SNAPSHOTS_DIR, filename)

@app.route('/analyzed_image')
def analyzed_image():
    import os
    img_path = "/dev/shm/analyzed.jpg"
    if os.path.exists(img_path):
        return send_file(img_path, mimetype="image/jpeg")
    else:
        frame = generate_mock_frame()
        cv2.putText(frame, "Waiting for YOLO analysis...", (50, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        _, jpeg = cv2.imencode('.jpg', frame)
        return Response(jpeg.tobytes(), mimetype="image/jpeg")

@app.route('/')
def index():
    return render_template('index.html')

# Stream video AI
@app.route('/video_feed')
def video_feed():
    return Response(video_stream_generator(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# API lấy dữ liệu cảm biến (cho frontend polling)
@app.route('/api/data', methods=['GET'])
def get_sensor_data():
    with data_lock:
        return jsonify(sensor_data)

# API điều khiển (Auto/Manual, Bật/Tắt Quạt/Phun sương)
@app.route('/api/control', methods=['POST'])
def post_control():
    global sensor_data, use_sensor_simulation
    req_data = request.json
    
    # Cập nhật cache local trước
    with data_lock:
        if "mode" in req_data:
            sensor_data["mode"] = req_data["mode"]
        if "fan" in req_data:
            sensor_data["fan"] = req_data["fan"]
        if "fogger" in req_data:
            sensor_data["fogger"] = req_data["fogger"]
            
    # Gửi lệnh trực tiếp sang ESP32-WROOM-32E nếu không phải chế độ giả lập
    if not use_sensor_simulation:
        try:
            ctrl_url = f"http://{ESP32_SENSOR_IP}/control"
            resp = requests.post(ctrl_url, json=req_data, timeout=1.5)
            if resp.status_code == 200:
                with data_lock:
                    sensor_data.update(resp.json())
                return jsonify(sensor_data)
        except Exception as e:
            print(f"Không thể gửi lệnh điều khiển đến ESP32: {e}")
            
    # Trả về trạng thái đã lưu trên cache
    with data_lock:
        return jsonify(sensor_data)

# API thiết lập thiết bị
@app.route('/api/settings', methods=['GET', 'POST'])
def handle_settings():
    global CAMERA_INDEX, ESP32_SENSOR_IP, use_cam_simulation, use_sensor_simulation
    
    if request.method == 'POST':
        req_data = request.json
        if "cam_ip" in req_data:
            try:
                # Đọc chỉ số camera index từ trường cam_ip gửi lên từ giao diện
                CAMERA_INDEX = int(req_data["cam_ip"])
            except ValueError:
                CAMERA_INDEX = 0
            use_cam_simulation = False
        if "sensor_ip" in req_data:
            ESP32_SENSOR_IP = req_data["sensor_ip"]
            use_sensor_simulation = False
            
        return jsonify({
            "status": "success",
            "cam_ip": str(CAMERA_INDEX),
            "sensor_ip": ESP32_SENSOR_IP
        })
    else:
        return jsonify({
            "cam_ip": str(CAMERA_INDEX),
            "sensor_ip": ESP32_SENSOR_IP,
            "cam_simulation": use_cam_simulation,
            "sensor_simulation": use_sensor_simulation
        })

if __name__ == '__main__':
    # Chạy server Flask trên cổng 5000, lắng nghe ở mọi interface mạng (0.0.0.0)
    app.run(host='0.0.0.0', port=5000, debug=False)
