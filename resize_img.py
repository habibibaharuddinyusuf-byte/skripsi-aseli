import os
from PIL import Image

# === Konfigurasi ===
input_folder = "dataset/muka/redness"     # Folder berisi gambar asli
output_folder = "resized/muka/redness"   # Folder hasil resize
new_size = (128, 128)             # Ukuran baru (lebar, tinggi)

# === Pastikan folder output ada ===
os.makedirs(output_folder, exist_ok=True)

# === Daftar ekstensi gambar yang didukung ===
valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

# === Proses resize ===
for filename in os.listdir(input_folder):
    if filename.lower().endswith(valid_extensions):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        try:
            with Image.open(input_path) as img:
                # Konversi ke RGB (mencegah error pada mode lain, misal PNG dengan alpha)
                img = img.convert("RGB")
                # Resize gambar
                img_resized = img.resize(new_size)
                # Simpan hasil
                img_resized.save(output_path)
                print(f"✅ {filename} -> {new_size}")
        except Exception as e:
            print(f"❌ Gagal memproses {filename}: {e}")

print("Selesai! Semua gambar sudah diresize.")
