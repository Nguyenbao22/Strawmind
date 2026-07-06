# StrawMind - Core System Package

Tài liệu này hướng dẫn cấu trúc các file cốt lõi của hệ thống AIoT giám sát nấm rơm và nhận diện bệnh bằng YOLOv8 Nano. Thư mục này chứa đầy đủ các tệp cần thiết nhất để chạy và phát triển dự án mà không bị lẫn các tệp dư thừa.

---

## 📁 Cấu trúc thư mục cốt lõi (Core Structure)

```text
StrawMind_Core/
│
├── deploy_to_pi.py          # Script tự động upload code và model từ PC lên Raspberry Pi 3
├── restart_and_monitor.py   # Script tự động khởi động lại dịch vụ trên Pi và kiểm tra log
│
├── 02_MODEL/
│   └── yolov8n_best.onnx    # Mô hình YOLOv8n đã cắt tỉa (Pruned) 320x320 dạng ONNX
│
├── 03_CODE/
│   ├── app.py               # Backend Flask Server (Xử lý web, nhận diện giả lập, lưu log bất đồng bộ)
│   ├── yolo_service.py      # Dịch vụ nhận diện YOLO thực tế chạy trên Pi (sử dụng OpenCV DNN để load ONNX)
│   ├── requirements.txt     # Danh sách thư viện Python cần thiết
│   │
│   ├── templates/
│   │   └── index.html       # Giao diện Web Dashboard (HTML5/CSS3/Vanilla JS phong cách nông nghiệp)
│   │
│   # Các mã nguồn phục vụ huấn luyện và tối ưu hóa (để tham khảo):
│   ├── prepare_dataset_v2.py # Script chuẩn bị dữ liệu và tự động gán nhãn tăng cường (Oversampling)
│   ├── train_yolov8n_320.py  # Script huấn luyện YOLOv8n ở độ phân giải 320x320
│   ├── prune_yolov8n.py      # Script cắt tỉa mô hình (L1-Pruning 30% Backbone & Neck)
│   └── quantize_yolov8n.py   # Script lượng tử hóa và xuất ONNX tương thích OpenCV DNN
│
├── 04_DATA/
│   ├── disease_logs.json    # File lưu lịch sử cảnh báo bệnh dạng JSON (Khởi tạo sẵn: [])
│   └── disease_snapshots/   # Thư mục lưu trữ ảnh chụp khoanh vùng nấm bị bệnh
│
# Cấu hình chạy giả lập cục bộ bằng Docker (để test trên PC):
├── Dockerfile
├── docker-compose.yml
├── run_dashboard.bat        # Click để chạy dashboard cục bộ bằng Docker
└── stop_dashboard.bat       # Click để tắt dashboard cục bộ
```

---

## 🚀 Hướng dẫn Vận hành & Phát triển

### 1. Chạy thử nghiệm trên Máy tính (Giả lập cảm biến và camera)
* **Cách 1 (Khuyên dùng)**: Mở Docker Desktop, click đúp file `run_dashboard.bat`. Sau đó truy cập `http://localhost:5000` trên trình duyệt.
* **Cách 2 (Chạy trực tiếp bằng Python)**:
  1. Di chuyển vào thư mục `03_CODE` và cài đặt thư viện: `pip install -r requirements.txt`
  2. Khởi chạy server: `python app.py`
  3. Truy cập `http://localhost:5000`. Dashboard sẽ tự động chạy ở chế độ giả lập thông số môi trường và camera.

### 2. Triển khai (Deploy) lên Raspberry Pi 3
Để đồng bộ hóa nhanh mã nguồn từ máy tính lên thiết bị Raspberry Pi 3 qua mạng Wi-Fi/LAN:
1. Mở file `deploy_to_pi.py` và cập nhật thông số kết nối ở dòng 24-26 (`pi_ip`, `pi_user`, `pi_pass`) cho đúng với mạng của bạn.
2. Chạy script deploy từ PC:
   ```bash
   python deploy_to_pi.py
   ```
3. Chạy script để tự động restart các dịch vụ (`strawmind.service` và `strawmind_yolo.service`) trên Pi và đọc log kiểm tra:
   ```bash
   python restart_and_monitor.py
   ```

### 3. Chạy bản cloud với 3 dịch vụ hiện tại

Thư mục `03_CODE` hiện có thêm file:

```text
app_cloud.py
```

File này là entrypoint mới để "cloud hóa" StrawMind Core mà không phá vỡ `app.py` cũ.

Vai trò của 3 cloud:

```text
HiveMQ Cloud
- nhận telemetry từ ESP32
- nhận/gửi lệnh điều khiển fan, fogger qua MQTT

Supabase
- lưu sensor_logs
- lưu mushroom_images
- lưu ai_detections
- lưu alerts

Render
- chạy Flask app_cloud.py
- render giao diện dashboard Core
- expose API upload ảnh /api/analyze
```

Luồng hệ thống cloud:

```text
ESP32 -> HiveMQ -> app_cloud.py -> Dashboard/Supabase
Pi AI hoặc client upload ảnh -> /api/analyze -> YOLO ONNX -> Dashboard/Supabase
Dashboard -> /api/control -> HiveMQ -> ESP32
```

### 4. Chạy bản cloud local trước khi deploy

1. Vào thư mục code:
   ```bash
   cd StrawMind_Core/03_CODE
   ```
2. Cài thư viện:
   ```bash
   pip install -r requirements.txt
   ```
3. Tạo env từ file mẫu:
   ```bash
   cp ../.env.cloud.example ../.env.cloud
   ```
4. Điền MQTT và Supabase key thật trong `../.env.cloud`
5. Export env rồi chạy:
   ```bash
   export $(grep -v '^#' ../.env.cloud | xargs)
   python app_cloud.py
   ```

Mở dashboard tại:

```text
http://localhost:5000
```

### 5. API quan trọng của bản cloud

Sensor/telemetry:

```text
MQTT topic nhận sensor: strawmind/<node_id>/telemetry
```

Điều khiển thiết bị:

```http
POST /api/control
```

Ví dụ body:

```json
{
  "mode": "manual",
  "fan": 1,
  "fogger": 0
}
```

Phân tích ảnh qua cloud:

```http
POST /api/analyze
```

Gửi `multipart/form-data` với field:

```text
image
```

API trạng thái:

```http
GET /api/status
GET /api/data
GET /api/disease_logs
```

### 6. Deploy bản cloud lên Render

Trên Render, tạo Web Service mới trỏ vào source này và chạy:

```text
Build command: pip install -r 03_CODE/requirements.txt
Start command: cd 03_CODE && python app_cloud.py
```

Environment variables cần thêm:

```text
PORT
NODE_ID
MODEL_PATH
MQTT_HOST
MQTT_PORT
MQTT_TOPIC_PREFIX
MQTT_USERNAME
MQTT_PASSWORD
SUPABASE_URL
SUPABASE_SERVICE_ROLE_KEY
```

Khuyến nghị:

```text
MODEL_PATH=/opt/render/project/src/02_MODEL/yolov8n_best.onnx
```

### 7. Phân tách rõ giữa bản local và bản cloud

```text
app.py
- dashboard local trên Pi/PC
- poll trực tiếp ESP qua HTTP
- dùng camera local /dev/shm

app_cloud.py
- dashboard cloud trên Render
- nhận sensor qua HiveMQ
- điều khiển ESP qua HiveMQ
- nhận ảnh qua /api/analyze
- lưu dữ liệu lên Supabase
```

---

## 💡 Điểm nổi bật kỹ thuật của bản cập nhật hiện tại:
* **Giao diện nông nghiệp tươi sáng & tương phản cao**: Sửa lỗi tương phản văn bản chỉ số trên các thẻ cảm biến (nhiệt độ đỏ đậm, độ ẩm xanh dương đậm, eCO2 xanh lá đậm trên nền pastel nhẹ).
* **Không treo Dashboard**: Các tiến trình ghi đĩa (sao chép ảnh YOLO cảnh báo sang SD card, ghi đè file JSON) đã được chạy bất đồng bộ bằng luồng phụ ngầm (`save_disease_log_async`), giúp ứng dụng chính của Flask phản hồi siêu nhanh mà không bị nghẽn I/O.
* **Modal Cấu hình & Chi tiết mượt mà**: Sửa lỗi Javascript sự kiện bấm "Cài đặt" (cho phép mở modal ngay lập tức) và phòng thủ lỗi `toFixed(1)` trong modal chi tiết nấm bệnh nếu dữ liệu cảm biến bị rỗng (`null`).
