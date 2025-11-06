"""
AUGMENTASI KONSERVATIF UNTUK GAMBAR MEDIS (PENYAKIT KULIT)
==========================================================

Prinsip: Jangan sampai mengubah karakteristik diagnostik penyakit!

AMAN untuk medical images:
✓ Rotasi ringan (max 10-15 derajat)
✓ Flip horizontal (jika penyakit tidak asimetris)
✓ Slight zoom/crop
✓ Minor brightness adjustment

HINDARI untuk medical images:
✗ Perubahan warna drastis (bisa ubah karakteristik lesi)
✗ Heavy blur (hilangkan detail penting)
✗ Distortion berat
✗ Extreme brightness/contrast
"""

import albumentations as A
import cv2
import os
from glob import glob
import numpy as np


# ============================================
# 1. AUGMENTASI SANGAT KONSERVATIF (PALING AMAN)
# ============================================
def get_ultra_conservative_transform():
    """
    Augmentasi paling minimal - hanya transformasi geometris dasar
    Cocok untuk: semua jenis penyakit kulit
    """
    transform = A.Compose([
        # Hanya rotasi kecil
        A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        
        # Flip horizontal (pastikan penyakitnya tidak asimetris!)
        A.HorizontalFlip(p=0.5),
        
        # Zoom/crop sedikit
        A.RandomResizedCrop(
            height=128, 
            width=128, 
            scale=(0.9, 1.0),  # Hanya zoom 90-100%
            p=0.5
        ),
        
        # Resize ke ukuran standard
        A.Resize(128, 128),
    ])
    
    return transform


# ============================================
# 2. AUGMENTASI KONSERVATIF (RECOMMENDED)
# ============================================
def get_conservative_transform():
    """
    Augmentasi konservatif dengan sedikit adjustment lighting
    Cocok untuk: kebanyakan kasus penyakit kulit
    """
    transform = A.Compose([
        # Transformasi geometris ringan
        A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05,   # Geser 5%
            scale_limit=0.1,    # Scale 10%
            rotate_limit=15,    # Rotasi 15 derajat
            border_mode=cv2.BORDER_CONSTANT,
            p=0.5
        ),
        
        # Brightness/contrast adjustment MINIMAL
        A.RandomBrightnessContrast(
            brightness_limit=0.1,  # Hanya +/- 10%
            contrast_limit=0.1,    # Hanya +/- 10%
            p=0.3
        ),
        
        # Resize
        A.Resize(128, 128),
    ])
    
    return transform


# ============================================
# 3. AUGMENTASI MODERAT (HATI-HATI)
# ============================================
def get_moderate_transform():
    """
    Augmentasi lebih agresif - gunakan dengan hati-hati
    Cocok untuk: dataset dengan lighting yang bervariasi
    """
    transform = A.Compose([
        # Transformasi geometris
        A.Rotate(limit=20, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.15,
            rotate_limit=20,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.5
        ),
        
        # Lighting adjustment moderat
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.4
        ),
        
        # Slight color adjustment (HATI-HATI!)
        A.HueSaturationValue(
            hue_shift_limit=5,      # Minimal hue shift
            sat_shift_limit=10,
            val_shift_limit=10,
            p=0.2
        ),
        
        # Sedikit noise untuk simulasi kualitas kamera berbeda
        A.GaussNoise(var_limit=(5.0, 15.0), p=0.1),
        
        # Resize
        A.Resize(128, 128),
    ])
    
    return transform


# ============================================
# 4. AUGMENTASI DENGAN PREVIEW
# ============================================
def augment_with_preview(image_path, num_samples=6, mode='conservative'):
    """
    Augmentasi dengan preview untuk melihat hasilnya
    """
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Pilih transform
    if mode == 'ultra_conservative':
        transform = get_ultra_conservative_transform()
    elif mode == 'conservative':
        transform = get_conservative_transform()
    else:
        transform = get_moderate_transform()
    
    # Create preview grid
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Augmentation Preview - Mode: {mode}', fontsize=16)
    
    for idx, ax in enumerate(axes.flat):
        if idx == 0:
            # Original image
            ax.imshow(cv2.resize(image, (128, 128)))
            ax.set_title('Original', fontsize=12, fontweight='bold')
        else:
            # Augmented images
            augmented = transform(image=image)
            aug_image = augmented['image']
            ax.imshow(aug_image)
            ax.set_title(f'Augmented {idx}', fontsize=12)
        
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('augmentation_preview.png', dpi=150, bbox_inches='tight')
    print("✓ Preview saved to: augmentation_preview.png")
    plt.close()


# ============================================
# 5. AUGMENTASI DATASET DENGAN MODE KONSERVATIF
# ============================================
def augment_medical_dataset(input_dir, output_dir, augmentations_per_image=3, mode='conservative'):
    """
    Augmentasi entire dataset dengan mode konservatif
    
    Args:
        input_dir: Folder berisi gambar original
        output_dir: Folder output
        augmentations_per_image: Jumlah variasi per gambar (3-5 recommended)
        mode: 'ultra_conservative', 'conservative', atau 'moderate'
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Pilih transform berdasarkan mode
    if mode == 'ultra_conservative':
        transform = get_ultra_conservative_transform()
        print("🔒 Mode: ULTRA CONSERVATIVE (Paling aman)")
    elif mode == 'conservative':
        transform = get_conservative_transform()
        print("✓ Mode: CONSERVATIVE (Recommended)")
    else:
        transform = get_moderate_transform()
        print("⚠️  Mode: MODERATE (Hati-hati, cek hasil manual)")
    
    # Get all image files
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []
    for ext in extensions:
        image_files.extend(glob(os.path.join(input_dir, ext)))
    
    if len(image_files) == 0:
        print(f"❌ Tidak ada gambar ditemukan di {input_dir}")
        return
    
    print(f"📁 Found {len(image_files)} images")
    print(f"🎯 Will generate {augmentations_per_image} augmentations per image")
    print(f"📊 Total output: {len(image_files) * (augmentations_per_image + 1)} images\n")
    
    # Process each image
    for idx, img_path in enumerate(image_files, 1):
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            print(f"⚠️  Skip: {os.path.basename(img_path)} (cannot read)")
            continue
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # Save original
        original_path = os.path.join(output_dir, f"{base_name}_original.jpg")
        # Resize original to standard size
        original_resized = cv2.resize(image, (128, 128))
        cv2.imwrite(original_path, original_resized)
        
        # Generate augmented versions
        for i in range(augmentations_per_image):
            try:
                augmented = transform(image=image_rgb)
                aug_image = augmented['image']
                
                # Save augmented image
                output_path = os.path.join(output_dir, f"{base_name}_aug{i+1}.jpg")
                aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path, aug_image_bgr)
            except Exception as e:
                print(f"⚠️  Error augmenting {base_name}: {e}")
        
        # Progress indicator
        if idx % 10 == 0 or idx == len(image_files):
            print(f"Progress: {idx}/{len(image_files)} images processed")
    
    total_output = len(glob(os.path.join(output_dir, "*.jpg")))
    print(f"\n✅ DONE! Total images created: {total_output}")
    print(f"📂 Output directory: {output_dir}")


# ============================================
# 6. QUALITY CHECK - Bandingkan Original vs Augmented
# ============================================
def quality_check_comparison(original_path, augmented_dir, num_samples=5):
    """
    Membandingkan gambar original dengan hasil augmentasi
    untuk memastikan karakteristik penyakit tidak berubah
    """
    import matplotlib.pyplot as plt
    
    # Load original
    original = cv2.imread(original_path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    original = cv2.resize(original, (128, 128))
    
    # Get augmented versions
    base_name = os.path.splitext(os.path.basename(original_path))[0]
    aug_files = sorted(glob(os.path.join(augmented_dir, f"{base_name}_aug*.jpg")))[:num_samples]
    
    if len(aug_files) == 0:
        print(f"❌ Tidak ada augmented images untuk {base_name}")
        return
    
    # Create comparison grid
    num_cols = len(aug_files) + 1
    fig, axes = plt.subplots(1, num_cols, figsize=(4*num_cols, 4))
    
    # Show original
    axes[0].imshow(original)
    axes[0].set_title('ORIGINAL', fontsize=14, fontweight='bold', color='blue')
    axes[0].axis('off')
    axes[0].set_facecolor('#e6f2ff')
    
    # Show augmented versions
    for idx, aug_path in enumerate(aug_files, 1):
        aug_img = cv2.imread(aug_path)
        aug_img = cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB)
        
        axes[idx].imshow(aug_img)
        axes[idx].set_title(f'Augmented {idx}', fontsize=12)
        axes[idx].axis('off')
    
    plt.suptitle(f'Quality Check: {base_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(augmented_dir, f"_quality_check_{base_name}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Quality check saved: {output_path}")
    plt.close()


# ============================================
# MAIN - CONTOH PENGGUNAAN
# ============================================
if __name__ == "__main__":
    # print("=" * 70)
    # print("AUGMENTASI KONSERVATIF UNTUK GAMBAR MEDIS (PENYAKIT KULIT)")
    # print("=" * 70)
    
    # print("\n📋 PANDUAN MEMILIH MODE:\n")
    
    # print("1. ULTRA CONSERVATIVE (Paling Aman)")
    # print("   - Hanya rotasi & flip")
    # print("   - Tidak mengubah warna/lighting sama sekali")
    # print("   - Gunakan untuk: Penyakit dengan karakteristik warna penting")
    # print("   - Contoh: Melanoma, Rosacea, Vitiligo\n")
    
    # print("2. CONSERVATIVE (Recommended) ⭐")
    # print("   - Rotasi, flip, zoom ringan")
    # print("   - Brightness/contrast adjustment minimal (10%)")
    # print("   - Gunakan untuk: Kebanyakan kasus penyakit kulit")
    # print("   - Contoh: Acne, Eczema, Dermatitis\n")
    
    # print("3. MODERATE (Hati-hati)")
    # print("   - Transformasi lebih agresif")
    # print("   - Slight color adjustment")
    # print("   - Gunakan untuk: Dataset dengan lighting bervariasi")
    # print("   - ⚠️  Wajib manual review hasil!\n")
    
    # print("=" * 70)
    # print("\n💡 CONTOH PENGGUNAAN:\n")
    
    print("# 1. Augmentasi dengan mode conservative (recommended)")
    augment_medical_dataset(
        input_dir='./resized/muka/bags',
        output_dir='./UC-aug/muka/bags',
        augmentations_per_image=15, 
        mode='conservative'
    )
    augment_medical_dataset(
        input_dir='./resized/muka/redness',
        output_dir='./UC-aug/muka/redness',
        augmentations_per_image=15, 
        mode='conservative'
    )
    
    print("# 2. Preview augmentation sebelum apply ke semua")
    # augment_with_preview('./resized/muka/acne/1.jpg', mode='conservative')
    
    print("# 3. Quality check hasil augmentasi")
    # quality_check_comparison('./resized/muka/acne/1.jpg', 'UC-aug/muka/acne', num_samples=5)
    
    print("\n" + "=" * 70)
    print("⚠️  CHECKLIST SEBELUM TRAINING:")
    print("=" * 70)
    print("☐ Manual review 10-20 augmented images")
    print("☐ Pastikan karakteristik penyakit tidak berubah")
    print("☐ Pastikan warna lesi masih natural")
    print("☐ Pastikan tekstur kulit masih terlihat")
    print("☐ Test dengan dokter/expert jika memungkinkan")
    print("=" * 70)