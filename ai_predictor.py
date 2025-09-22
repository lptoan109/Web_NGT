import os
import numpy as np
import noisereduce as nr
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
import audio_utils_light as audioimport

# Các hằng số cho việc xử lý âm thanh
SAMPLE_RATE = 16000
MIN_DURATION_S = 1.0
SILENCE_THRESHOLD_DB = 40

# Các hằng số cho Mel Spectrogram
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128

class CoughPredictor:
    def __init__(self, model_path):
        # --- THAY ĐỔI LỚN: Tải mô hình Keras đầy đủ ---
        self.model = tf.keras.models.load_model(model_path)
        # -----------------------------------------------
        self.labels = ["asthma", "covid", "healthy", "tuberculosis"]

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def _process_audio(self, file_path):
        try:
            y, sr = sf.read(file_path, dtype='float32')
            if sr != SAMPLE_RATE:
                print(f"Cảnh báo: Tần số lấy mẫu không phải {SAMPLE_RATE}Hz.")
            y = audio.to_mono(y)
            if len(y) / sr < MIN_DURATION_S: return None
            y = audio.normalize(y)
            y_denoised = nr.reduce_noise(y=y, sr=sr)
            y_trimmed = audio.trim(y_denoised, top_db=SILENCE_THRESHOLD_DB)
            if len(y_trimmed) < N_FFT: return None
            mel_spec = audio.melspectrogram(y=y_trimmed, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
            db_spec = audio.power_to_db(mel_spec, top_db=80)
            spec_rgb = np.stack([db_spec, db_spec, db_spec], axis=-1)
            spec_expanded = np.expand_dims(spec_rgb, axis=0)
            preprocessed_spec = preprocess_input(spec_expanded)
            return preprocessed_spec
        except Exception as e:
            print(f"Lỗi khi xử lý file {os.path.basename(file_path)}: {e}")
            return None

    def predict(self, audio_file_path):
        input_tensor = self._process_audio(audio_file_path)
        if input_tensor is None:
            return {"error": "Không thể xử lý file âm thanh (quá ngắn hoặc lỗi)."}

        # --- THAY ĐỔI LỚN: Chạy dự đoán bằng model.predict ---
        output_data = self.model.predict(input_tensor)
        # ---------------------------------------------------
        
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