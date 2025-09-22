# ai_predictor.py (phiên bản cuối cùng, an toàn với Padding & Cropping)

import os
import numpy as np
import noisereduce as nr
import soundfile as sf
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
import audio_utils_light as audio

# Các hằng số
SAMPLE_RATE = 16000
MIN_DURATION_S = 1.0
SILENCE_THRESHOLD_DB = 40
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
TARGET_SHAPE = (240, 240) # Chiều cao, chiều rộng mong muốn

class CoughPredictor:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        self.labels = ["asthma", "covid", "healthy", "tuberculosis"]

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def _process_audio(self, file_path):
        try:
            y, sr = sf.read(file_path, dtype='float32')
            if sr != SAMPLE_RATE: print(f"Cảnh báo: Tần số lấy mẫu không phải {SAMPLE_RATE}Hz.")
            y = audio.to_mono(y)
            if len(y) / sr < MIN_DURATION_S: return None
            y = audio.normalize(y)
            y_denoised = nr.reduce_noise(y=y, sr=sr)
            y_trimmed = audio.trim(y_denoised, top_db=SILENCE_THRESHOLD_DB)
            if len(y_trimmed) < N_FFT: return None
            
            mel_spec = audio.melspectrogram(y=y_trimmed, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
            db_spec = audio.power_to_db(mel_spec, top_db=80)

            # --- THAY ĐỔI LỚN: XỬ LÝ KÍCH THƯỚC AN TOÀN ---
            # 1. Lấy kích thước hiện tại
            current_height, current_width = db_spec.shape
            target_height, target_width = TARGET_SHAPE

            # 2. Xử lý chiều rộng (thời gian)
            if current_width < target_width:
                # Nếu ngắn hơn -> Đệm (Padding) bằng số 0
                pad_width = target_width - current_width
                # Đệm vào bên phải của spectrogram
                padded_spec = np.pad(db_spec, ((0, 0), (0, pad_width)), mode='constant', constant_values=np.min(db_spec))
            else:
                # Nếu dài hơn -> Cắt (Cropping) ở giữa
                start = (current_width - target_width) // 2
                padded_spec = db_spec[:, start:start + target_width]
            
            # 3. Xử lý chiều cao (số Mel bins) - Thường không cần nhưng để đảm bảo
            if current_height != target_height:
                 # Nếu chiều cao không khớp, dùng resize vì số Mel bins là cố định
                 padded_spec = tf.image.resize(np.expand_dims(padded_spec, axis=-1), [target_height, target_width]).numpy().squeeze()

            # Spectrogram cuối cùng đã có kích thước 240x240
            final_spec = padded_spec
            # ---------------------------------------------------

            spec_rgb = np.stack([final_spec, final_spec, final_spec], axis=-1)
            spec_expanded = np.expand_dims(spec_rgb, axis=0)
            preprocessed_spec = preprocess_input(spec_expanded)
            return preprocessed_spec
        except Exception as e:
            print(f"Lỗi khi xử lý file {os.path.basename(file_path)}: {e}")
            return None

    def predict(self, audio_file_path):
        # Hàm này không cần thay đổi
        input_tensor = self._process_audio(audio_file_path)
        if input_tensor is None:
            return {"error": "Không thể xử lý file âm thanh (quá ngắn hoặc lỗi)."}

        output_data = self.model.predict(input_tensor)
        
        predictions_logits = output_data[0]
        prediction_probs = self._softmax(predictions_logits)
        
        predicted_class_index = np.argmax(prediction_probs)
        predicted_class_name = self.labels[predicted_class_index]
        confidence = float(prediction_probs[predicted_class_index])
        
        display_names = {"healthy": "Khỏe mạnh", "asthma": "Hen suyễn", "covid": "COVID-19", "tuberculosis": "Lao"}

        return {
            "predicted_class": display_names.get(predicted_class_name, predicted_class_name),
            "confidence": f"{confidence:.2%}",
            "details": {display_names.get(label, label): f"{prob:.2%}" for label, prob in zip(self.labels, prediction_probs)}
        }