#!/usr/bin/env python3
"""
🖼️ PREPROCESSING SCRIPT: Standardize Inconsistent Photos

Purpose:
- Standardize photos dengan berbagai angle/fokus
- Crop ke area face/skin yang relevant
- Normalize lighting dan color
- Resize ke ukuran konsisten

Usage:
    python preprocess_photos.py --input ./raw_dataset --output ./processed_dataset
"""

import os
import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import shutil
from PIL import Image, ImageEnhance
import json

class PhotoStandardizer:
    """
    Standardize photos untuk skin disease classification.
    
    Features:
    1. Face detection & crop
    2. Lighting normalization
    3. Color correction
    4. Resize to standard size
    5. Quality filtering
    """
    
    def __init__(self, target_size=(96, 96), enable_face_detection=True):
        self.target_size = target_size
        self.enable_face_detection = enable_face_detection
        
        # Load face detector
        if enable_face_detection:
            try:
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
                print("✅ Face detector loaded")
            except Exception as e:
                print(f"⚠️ Face detector not available: {e}")
                self.enable_face_detection = False
        
        # Statistics untuk reporting
        self.stats = {
            'total': 0,
            'success': 0,
            'face_detected': 0,
            'center_crop': 0,
            'failed': 0,
            'low_quality': 0
        }
    
    def detect_and_crop_face(self, image):
        """
        Detect face dan crop ke area face.
        
        Returns:
            cropped_image, method_used
        """
        # Convert to grayscale for detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) > 0:
            # Get largest face
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = largest_face
            
            # Add padding (20% on each side)
            padding = int(max(w, h) * 0.2)
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(image.shape[1], x + w + padding)
            y2 = min(image.shape[0], y + h + padding)
            
            cropped = image[y1:y2, x1:x2]
            
            self.stats['face_detected'] += 1
            return cropped, 'face_detection'
        else:
            # Fallback: center crop
            return self.center_crop_square(image), 'center_crop'
    
    def center_crop_square(self, image):
        """
        Center crop image to square.
        """
        h, w = image.shape[:2]
        size = min(h, w)
        
        start_h = (h - size) // 2
        start_w = (w - size) // 2
        
        cropped = image[start_h:start_h+size, start_w:start_w+size]
        
        self.stats['center_crop'] += 1
        return cropped
    
    def normalize_lighting(self, image):
        """
        Normalize lighting menggunakan CLAHE (Contrast Limited Adaptive Histogram Equalization).
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Split channels
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels
        lab = cv2.merge([l, a, b])
        
        # Convert back to BGR
        normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return normalized
    
    def color_correction(self, image):
        """
        Simple color correction untuk skin tone.
        """
        # Convert to float
        img_float = image.astype(np.float32) / 255.0
        
        # Apply gamma correction (brighten slightly)
        gamma = 1.2
        img_corrected = np.power(img_float, 1/gamma)
        
        # Enhance saturation slightly
        img_hsv = cv2.cvtColor(img_corrected, cv2.COLOR_BGR2HSV)
        img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1] * 1.1, 0, 1)
        img_corrected = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
        
        # Convert back to uint8
        img_corrected = (img_corrected * 255).astype(np.uint8)
        
        return img_corrected
    
    def check_quality(self, image, min_size=50, max_blur=100):
        """
        Check image quality.
        
        Returns:
            is_good, reason
        """
        # Check size
        h, w = image.shape[:2]
        if h < min_size or w < min_size:
            return False, f"too_small_{w}x{h}"
        
        # Check blur (using Laplacian variance)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if laplacian_var < max_blur:
            return False, f"too_blurry_{laplacian_var:.1f}"
        
        # Check if mostly black (underexposed)
        if np.mean(gray) < 20:
            return False, "too_dark"
        
        # Check if mostly white (overexposed)
        if np.mean(gray) > 235:
            return False, "too_bright"
        
        return True, "good"
    
    def process_image(self, input_path, output_path, save_comparison=False):
        """
        Process single image dengan semua preprocessing steps.
        
        Returns:
            success, message
        """
        self.stats['total'] += 1
        
        try:
            # Read image
            image = cv2.imread(input_path)
            
            if image is None:
                self.stats['failed'] += 1
                return False, "cannot_read"
            
            original = image.copy()
            
            # Check quality
            is_good, reason = self.check_quality(image)
            if not is_good:
                self.stats['low_quality'] += 1
                return False, f"low_quality_{reason}"
            
            # Step 1: Crop (face detection or center crop)
            if self.enable_face_detection:
                cropped, method = self.detect_and_crop_face(image)
            else:
                cropped = self.center_crop_square(image)
                method = 'center_crop'
            
            # Step 2: Normalize lighting
            normalized = self.normalize_lighting(cropped)
            
            # Step 3: Color correction
            corrected = self.color_correction(normalized)
            
            # Step 4: Resize to target size
            final = cv2.resize(corrected, self.target_size, interpolation=cv2.INTER_LANCZOS4)
            
            # Save processed image
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, final)
            
            # Save comparison (optional)
            if save_comparison:
                comparison_path = output_path.replace('.jpg', '_comparison.jpg')
                self.save_comparison(original, final, comparison_path)
            
            self.stats['success'] += 1
            return True, method
            
        except Exception as e:
            self.stats['failed'] += 1
            return False, str(e)
    
    def save_comparison(self, original, processed, output_path):
        """
        Save side-by-side comparison.
        """
        # Resize original untuk comparison
        h, w = processed.shape[:2]
        original_resized = cv2.resize(original, (w, h))
        
        # Concatenate
        comparison = np.hstack([original_resized, processed])
        
        # Add labels
        cv2.putText(comparison, "Original", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(comparison, "Processed", (w + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imwrite(output_path, comparison)
    
    def print_stats(self):
        """
        Print processing statistics.
        """
        print("\n" + "="*60)
        print("📊 PREPROCESSING STATISTICS")
        print("="*60)
        print(f"Total images: {self.stats['total']}")
        print(f"✅ Successfully processed: {self.stats['success']}")
        print(f"   - Face detected: {self.stats['face_detected']}")
        print(f"   - Center cropped: {self.stats['center_crop']}")
        print(f"❌ Failed: {self.stats['failed']}")
        print(f"⚠️  Low quality: {self.stats['low_quality']}")
        print(f"Success rate: {self.stats['success']/self.stats['total']*100:.1f}%")
        print("="*60)


def process_dataset(input_dir, output_dir, target_size=(96, 96), 
                   save_comparison=False, face_detection=True):
    """
    Process entire dataset directory.
    
    Directory structure:
        input_dir/
        ├── Class1/
        │   ├── img1.jpg
        │   └── img2.jpg
        └── Class2/
            └── img3.jpg
    
    Output structure will be the same.
    """
    print("\n🚀 Starting dataset preprocessing...")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Target size: {target_size}")
    print(f"Face detection: {face_detection}")
    print()
    
    # Initialize preprocessor
    preprocessor = PhotoStandardizer(
        target_size=target_size,
        enable_face_detection=face_detection
    )
    
    # Get all class directories
    class_dirs = [d for d in os.listdir(input_dir) 
                  if os.path.isdir(os.path.join(input_dir, d))]
    
    print(f"Found {len(class_dirs)} classes:")
    for cls in class_dirs:
        print(f"  - {cls}")
    print()
    
    # Process each class
    failed_images = []
    
    for class_name in class_dirs:
        print(f"\n📂 Processing class: {class_name}")
        
        input_class_dir = os.path.join(input_dir, class_name)
        output_class_dir = os.path.join(output_dir, class_name)
        
        # Get all images
        image_files = [f for f in os.listdir(input_class_dir)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        print(f"   Found {len(image_files)} images")
        
        # Process with progress bar
        for img_file in tqdm(image_files, desc=f"   {class_name}"):
            input_path = os.path.join(input_class_dir, img_file)
            output_path = os.path.join(output_class_dir, img_file)
            
            # Change extension to .jpg
            if not output_path.lower().endswith('.jpg'):
                output_path = os.path.splitext(output_path)[0] + '.jpg'
            
            success, message = preprocessor.process_image(
                input_path, output_path, save_comparison=save_comparison
            )
            
            if not success:
                failed_images.append({
                    'class': class_name,
                    'file': img_file,
                    'reason': message
                })
    
    # Print summary
    preprocessor.print_stats()
    
    # Save failed images log
    if failed_images:
        print(f"\n⚠️  {len(failed_images)} images failed processing:")
        
        failed_log_path = os.path.join(output_dir, 'failed_images.json')
        with open(failed_log_path, 'w') as f:
            json.dump(failed_images, f, indent=2)
        
        print(f"   Failed images log saved: {failed_log_path}")
        
        # Print first few
        for fail in failed_images[:5]:
            print(f"   - {fail['class']}/{fail['file']}: {fail['reason']}")
        
        if len(failed_images) > 5:
            print(f"   ... and {len(failed_images) - 5} more")
    
    # Save statistics
    stats_path = os.path.join(output_dir, 'preprocessing_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(preprocessor.stats, f, indent=2)
    
    print(f"\n💾 Statistics saved: {stats_path}")
    
    print("\n✅ Preprocessing completed!")


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess skin disease images for consistent quality'
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input directory containing class folders'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output directory for processed images'
    )
    
    parser.add_argument(
        '--size', '-s',
        type=int,
        default=96,
        help='Target image size (default: 96)'
    )
    
    parser.add_argument(
        '--no-face-detection',
        action='store_true',
        help='Disable face detection (use center crop only)'
    )
    
    parser.add_argument(
        '--save-comparison',
        action='store_true',
        help='Save before/after comparison images'
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.input):
        print(f"❌ Input directory not found: {args.input}")
        return
    
    # Process dataset
    process_dataset(
        input_dir=args.input,
        output_dir=args.output,
        target_size=(args.size, args.size),
        save_comparison=args.save_comparison,
        face_detection=not args.no_face_detection
    )


if __name__ == '__main__':
    main()


# ========================================
# USAGE EXAMPLES
# ========================================

"""
# Basic usage:
python preprocess_photos.py --input ./raw_dataset --output ./processed_dataset

# With custom size:
python preprocess_photos.py --input ./raw_dataset --output ./processed_dataset --size 128

# Disable face detection (faster, but less accurate):
python preprocess_photos.py --input ./raw_dataset --output ./processed_dataset --no-face-detection

# Save comparison images (untuk visual inspection):
python preprocess_photos.py --input ./raw_dataset --output ./processed_dataset --save-comparison

# Full example:
python preprocess_photos.py \
    --input ./dataset/raw \
    --output ./dataset/processed \
    --size 96 \
    --save-comparison
"""
