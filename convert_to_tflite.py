import tensorflow as tf
import os
import glob
import time

print(f"Đang sử dụng TensorFlow phiên bản: {tf.__version__}")
print("="*50)

# --- CẤU HÌNH ---
# Đường dẫn đến thư mục chứa 5 model .keras
# (Giữ nguyên đường dẫn này vì nó đã đúng)
SOURCE_MODEL_DIR = "Ngt-Cough-Api/models" 

# Đường dẫn đến thư mục để lưu 5 model .tflite mới
DEST_TFLITE_DIR = "models_tflite" 
# ------------------

# Lấy đường dẫn tuyệt đối để gỡ lỗi
abs_source_dir = os.path.abspath(SOURCE_MODEL_DIR)
abs_dest_dir = os.path.abspath(DEST_TFLITE_DIR)

print(f"Thư mục nguồn (tuyệt đối): {abs_source_dir}")
print(f"Thư mục đích (tuyệt đối):  {abs_dest_dir}")
print("="*50)

os.makedirs(DEST_TFLITE_DIR, exist_ok=True)

# Tìm tất cả các file .keras trong thư mục nguồn
keras_model_paths = glob.glob(os.path.join(SOURCE_MODEL_DIR, "*.keras"))

if not keras_model_paths:
    print(f"LỖI: Không tìm thấy file .keras nào trong thư mục '{SOURCE_MODEL_DIR}'")
    print("Vui lòng kiểm tra lại đường dẫn SOURCE_MODEL_DIR.")
    exit()

print(f"Tìm thấy {len(keras_model_paths)} mô hình .keras để chuyển đổi...")
success_count = 0
fail_count = 0

# Lặp qua từng model để chuyển đổi
for model_path in keras_model_paths:
    print(f"\n--- BẮT ĐẦU CHUYỂN ĐỔI: {model_path} ---")
    model_name = os.path.basename(model_path)
    tflite_name = model_name.replace(".keras", ".tflite")
    dest_path = os.path.join(DEST_TFLITE_DIR, tflite_name)
    
    try:
        # 1. Tải mô hình
        print(f"[1/5] Đang tải mô hình: {model_path}")
        model = tf.keras.models.load_model(model_path)
        print("[1/5] Tải mô hình thành công.")
        
        # 2. Khởi tạo bộ chuyển đổi
        print("[2/5] Đang khởi tạo bộ chuyển đổi TFLite...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        print("[2/5] Khởi tạo thành công.")
        
        # 3. Chuyển đổi (Đây là bước có thể mất thời gian)
        print("[3/5] Đang tiến hành lượng tử hóa (convert)... Vui lòng chờ...")
        tflite_model = converter.convert()
        print("[3/5] Lượng tử hóa thành công.")
        
        # 4. Lưu file
        print(f"[4/5] Đang lưu mô hình TFLite tại: {dest_path}")
        with open(dest_path, 'wb') as f:
            f.write(tflite_model)
        print("[4/5] Lưu file thành công.")
        
        # 5. Kiểm tra file
        print(f"[5/5] Đang kiểm tra file đã lưu...")
        time.sleep(1) # Đợi 1 giây để hệ thống file cập nhật
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
            print(f"*** THÀNH CÔNG! Đã tạo file: {tflite_name} (Kích thước: {os.path.getsize(dest_path)} bytes) ***")
            success_count += 1
        else:
            print(f"*** LỖI NGHIÊM TRỌNG: File KHÔNG được tạo hoặc rỗng tại {dest_path} ***")
            fail_count += 1
            
    except Exception as e:
        print(f"!!! LỖI TOÀN BỘ khi chuyển đổi {model_path}: {e} !!!")
        fail_count += 1
        # Tiếp tục với file tiếp theo
        continue

print("\n" + "="*50)
print("--- QUÁ TRÌNH HOÀN TẤT! ---")
print(f"Thành công: {success_count} file(s)")
print(f"Thất bại:  {fail_count} file(s)")
print(f"Các mô hình TFLite (nếu có) đã được lưu trong thư mục: '{DEST_TFLITE_DIR}'")
print("="*50)