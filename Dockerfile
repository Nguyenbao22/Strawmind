# Sử dụng base image Python chính thức (nhẹ và ổn định)
FROM python:3.10-slim

# Thiết lập thư mục làm việc trong container
WORKDIR /app

# Cài đặt các thư viện hệ thống cần thiết cho OpenCV và YOLOv8
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt PyTorch phiên bản CPU để tiết kiệm bộ nhớ (tránh lỗi cạn kiệt tài nguyên)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Sao chép file requirements.txt
COPY 03_CODE/requirements.txt .

# Cài đặt các thư viện Python
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép toàn bộ dự án vào container (bao gồm cả thư mục 02_MODEL và 03_CODE)
COPY . .

# Expose cổng 5000 (cổng chạy Flask)
EXPOSE 5000

# Chuyển vào thư mục 03_CODE làm việc để chạy app.py
WORKDIR /app/03_CODE

# Khởi chạy ứng dụng
CMD ["python", "app.py"]
