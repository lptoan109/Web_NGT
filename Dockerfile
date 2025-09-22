# Dockerfile

# Bắt đầu từ một hình ảnh Docker chính thức của TensorFlow
# Nó đã có sẵn Python, TensorFlow và tất cả các thư viện Flex Delegate cần thiết.
FROM tensorflow/tensorflow:2.15.0

# Thiết lập thư mục làm việc bên trong container
WORKDIR /app

# Sao chép file requirements.txt vào trước để tận dụng cache của Docker
COPY requirements.txt .

# Chạy lệnh pip install để cài đặt các thư viện còn lại
# TensorFlow đã có sẵn nên không cần cài lại
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép toàn bộ mã nguồn của bạn vào container
COPY . .

# Lệnh để chạy ứng dụng của bạn khi container khởi động
# Gunicorn sẽ lắng nghe trên cổng 10000, Render sẽ tự động biết điều này
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "app:app"]