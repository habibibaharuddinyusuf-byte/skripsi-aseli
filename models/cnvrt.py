import tensorflow as tf
import numpy as np

# === 1️⃣ Muat model (.keras) dengan Lambda layer ===
model = tf.keras.models.load_model("retinal_disease_model.keras", safe_mode=False)

# === 2️⃣ Bangun model (agar punya input signature) ===
# Sesuai dengan input shape model kamu: (224, 224, 3)
model.build((None, 224, 224, 3))
# atau panggil dummy data agar semua layer siap
_ = model(np.random.rand(1, 224, 224, 3).astype(np.float32))

# === 3️⃣ Ekspor ke format SavedModel (ganti .export()) ===
model.export("saved_model_dir")

# === 4️⃣ Konversi ke TensorFlow Lite ===
converter = tf.lite.TFLiteConverter.from_saved_model("saved_model_dir")

# Tidak melakukan quantization (float32 penuh, akurasi 100%)
tflite_model = converter.convert()

# === 5️⃣ Simpan model TFLite ===
with open("model_mobile_float32.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Konversi sukses! Model tersimpan sebagai 'model_mobile_float32.tflite'")
