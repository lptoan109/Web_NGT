# Dockerfile (phiên bản cuối cùng, đã sửa lỗi tìm kiếm)

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
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# --- THAY ĐỔI QUAN TRỌNG NHẤT ---
# Tìm đường dẫn chính xác đến thư viện libtensorflowlite_flex.so trên TOÀN BỘ HỆ THỐNG
# và thêm nó vào biến môi trường LD_LIBRARY_PATH để hệ thống có thể tìm thấy.
RUN TFLITE_FLEX_PATH=$(find / -name "libtensorflowlite_flex.so" | head -n 1 | xargs dirname) && \
    echo "Found Flex Delegate at: $TFLITE_FLEX_PATH" && \
    echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TFLITE_FLEX_PATH" >> /etc/profile
# --------------------------------

# Sao chép toàn bộ mã nguồn của bạn vào container
COPY . .

# Lệnh để chạy ứng dụng của bạn khi container khởi động
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "app:app"]