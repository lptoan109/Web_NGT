# Dockerfile (phiên bản cuối cùng, đơn giản và hiệu quả)

# Bắt đầu từ hình ảnh Docker chính thức của TensorFlow
FROM tensorflow/tensorflow:2.15.0

# Tạo một môi trường ảo tại /opt/venv
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
# Thêm thư mục bin của môi trường ảo vào PATH
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Thiết lập thư mục làm việc bên trong container
WORKDIR /app

# Sao chép file requirements.txt vào trước
COPY requirements.txt .

# Nâng cấp pip và cài đặt thư viện vào môi trường ảo
# Việc cài đặt tensorflow-text sẽ tự động giải quyết vấn đề libtensorflowlite_flex.so
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép toàn bộ mã nguồn của bạn vào container
COPY . .

# Lệnh để chạy ứng dụng của bạn khi container khởi động
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "app:app"]