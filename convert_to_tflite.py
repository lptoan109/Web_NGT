import tensorflow as tf
import os
import glob
import time

# =============================================================================
# BƯỚC 1: CODE CUSTOM TỪ NOTEBOOK CỦA BẠN (GIỮ LẠI DECORATOR)
# =============================================================================

@tf.keras.saving.register_keras_serializable(package="Custom")
class FinalModelCNN(tf.keras.Model):
    def __init__(self, input_shape_config, num_classes_config, **kwargs):
        super(FinalModelCNN, self).__init__(**kwargs)
        self.input_shape_config = input_shape_config
        self.num_classes_config = num_classes_config

        # 1. Base Model (CNN)
        self.base_model = tf.keras.applications.EfficientNetV2B2(
            weights='imagenet', 
            include_top=False, 
            input_shape=self.input_shape_config
        )
        
        # 2. Lớp Pooling để giảm chiều dữ liệu
        self.gap = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")
        
        # 3. Các lớp Dense (Head)
        self.dense1 = tf.keras.layers.Dense(512, use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(1e-5), name="dense_layer_1")
        self.bn1 = tf.keras.layers.BatchNormalization(name="batch_norm_1")
        self.act1 = tf.keras.layers.Activation('relu', name="activation_1")
        self.dropout1 = tf.keras.layers.Dropout(0.3, name="dropout_layer_1")
        
        self.dense2 = tf.keras.layers.Dense(256, use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(1e-5), name="dense_layer_2")
        self.bn2 = tf.keras.layers.BatchNormalization(name="batch_norm_2")
        self.act2 = tf.keras.layers.Activation('relu', name="activation_2")
        self.dropout2 = tf.keras.layers.Dropout(0.2, name="dropout_layer_2")
        
        # 4. Lớp Output
        self.dense_output = tf.keras.layers.Dense(self.num_classes_config, activation='linear', dtype='float32', name="output_layer")

    def call(self, inputs, training=None):
        x = self.base_model(inputs, training=training)
        x = self.gap(x, training=training) 
        
        x = self.dense1(x)
        x = self.bn1(x, training=training); x = self.act1(x); x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.bn2(x, training=training); x = self.act2(x); x = self.dropout2(x, training=training)
        outputs = self.dense_output(x)
        return outputs

    def get_config(self):
        config = super(FinalModelCNN, self).get_config()
        config.update({
            'input_shape_config': self.input_shape_config,
            'num_classes_config': self.num_classes_config,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

@tf.keras.saving.register_keras_serializable(package="Custom")
class MacroF1Score(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name='f1_macro', **kwargs):
        super(MacroF1Score, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.true_positives = self.add_weight(name='tp', shape=(num_classes,), initializer='zeros')
        self.false_positives = self.add_weight(name='fp', shape=(num_classes,), initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', shape=(num_classes,), initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_labels = tf.argmax(tf.nn.softmax(y_pred), axis=1)
        y_true_labels = tf.argmax(y_true, axis=1)
        cm = tf.math.confusion_matrix(y_true_labels, y_pred_labels, num_classes=self.num_classes, dtype=tf.float32)
        tp = tf.linalg.diag_part(cm)
        fp = tf.reduce_sum(cm, axis=0) - tp
        fn = tf.reduce_sum(cm, axis=1) - tp
        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)

    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + tf.keras.backend.epsilon())
        recall = self.true_positives / (self.true_positives + self.false_negatives + tf.keras.backend.epsilon())
        f1 = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
        macro_f1 = tf.reduce_mean(f1)
        return macro_f1

    def reset_state(self):
        self.true_positives.assign(tf.zeros(self.num_classes))
        self.false_positives.assign(tf.zeros(self.num_classes))
        self.false_negatives.assign(tf.zeros(self.num_classes))

    def get_config(self):
        config = super(MacroF1Score, self).get_config()
        config.update({'num_classes': self.num_classes})
        return config

# =============================================================================
# KẾT THÚC PHẦN CODE CUSTOM
# =============================================================================


print(f"Đang sử dụng TensorFlow phiên bản: {tf.__version__}")
print("="*50)

# --- CẤU HÌNH ---
SOURCE_MODEL_DIR = "models" 
DEST_TFLITE_DIR = "models_tflite" 
# ------------------

abs_source_dir = os.path.abspath(SOURCE_MODEL_DIR)
abs_dest_dir = os.path.abspath(DEST_TFLITE_DIR)

print(f"Thư mục nguồn (tuyệt đối): {abs_source_dir}")
print(f"Thư mục đích (tuyệt đối):  {abs_dest_dir}")
print("="*50)

os.makedirs(DEST_TFLITE_DIR, exist_ok=True)

# === TEST 1 MODEL THỬ NGHIỆM ===
MODEL_TEST_NAME = "CNN_from_NPY_no_val.keras"
model_test_path = os.path.join(SOURCE_MODEL_DIR, MODEL_TEST_NAME)
keras_model_paths = [model_test_path]
print(f"!!! CHẾ ĐỘ TEST: CHỈ CHUYỂN ĐỔI 1 MÔ HÌNH ({MODEL_TEST_NAME}) !!!")

if not os.path.exists(keras_model_paths[0]):
    print(f"LỖI: Không tìm thấy file model mẫu: {keras_model_paths[0]}")
    exit()

print(f"Tìm thấy {len(keras_model_paths)} mô hình .keras để chuyển đổi...")
success_count = 0
fail_count = 0

# =============================================================================
# === SỬA LỖI QUAN TRỌNG: Dùng tên đã đăng ký làm 'key' ===
# =============================================================================
# Dựa trên log lỗi, Keras đang tìm các 'key' này:
custom_objects_dict = {
    "Custom>FinalModelCNN": FinalModelCNN,
    "Custom>MacroF1Score": MacroF1Score
}
# =============================================================================

for model_path in keras_model_paths:
    print(f"\n--- BẮT ĐẦU CHUYỂN ĐỔI: {model_path} ---")
    model_name = os.path.basename(model_path)
    tflite_name = model_name.replace(".keras", ".tflite")
    dest_path = os.path.join(DEST_TFLITE_DIR, tflite_name)
    
    try:
        # Tải mô hình với 'custom_objects' đã sửa
        print(f"[1/5] Đang tải mô hình (load_model) với custom objects: {model_path}")
        
        model = tf.keras.models.load_model(
            model_path, 
            custom_objects=custom_objects_dict,
            compile=False  # Vẫn giữ 'compile=False' để bỏ qua optimizer
        )
        
        print("[1/5] Tải mô hình thành công.")
        
        # 2. Khởi tạo bộ chuyển đổi
        print("[2/5] Đang khởi tạo bộ chuyển đổi TFLite...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        print("[2/5] Khởi tạo thành công.")
        
        # 3. Chuyển đổi
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
        time.sleep(1) 
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
            print(f"*** THÀNH CÔNG! Đã tạo file: {tflite_name} (Kích thước: {os.path.getsize(dest_path)} bytes) ***")
            success_count += 1
        else:
            print(f"*** LỖI NGHIÊM TRỌNG: File KHÔNG được tạo hoặc rỗng tại {dest_path} ***")
            fail_count += 1
            
    except Exception as e:
        print(f"!!! LỖI TOÀN BỘ khi chuyển đổi {model_path}: {e} !!!")
        fail_count += 1
        continue

print("\n" + "="*50)
print("--- QUÁ TRÌNH HOÀN TẤT! ---")
print(f"Thành công: {success_count} file(s)")
print(f"Thất bại:  {fail_count} file(s)")
print(f"Các mô hình TFLite (nếu có) đã được lưu trong thư mục: '{DEST_TFLITE_DIR}'")
print("="*50)