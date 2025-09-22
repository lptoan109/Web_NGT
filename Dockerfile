# Dockerfile (phiên bản cuối cùng, đã bao gồm gói hệ thống cần thiết)

# Bắt đầu từ một hình ảnh Docker Python 3.10 chính thức, sạch sẽ
FROM python:3.10-slim

# Thiết lập các biến môi trường để Python không tạo file .pyc
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Tạo và chuyển đến thư mục làm việc
WORKDIR /app

# --- THÊM VÀO: Cài đặt gói hệ thống cần thiết cho việc xử lý âm thanh ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
# ---------------------------------------------------------------------

# Nâng cấp pip
RUN pip install --upgrade pip

# Sao chép file requirements.txt
COPY requirements.txt .

# Cài đặt TẤT CẢ các thư viện Python từ requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép toàn bộ mã nguồn của bạn vào container
COPY . .

# Lệnh để chạy ứng dụng của bạn khi container khởi động
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "app:app"]