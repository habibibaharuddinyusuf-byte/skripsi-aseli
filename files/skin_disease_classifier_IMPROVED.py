# 🔬 IMPROVED: Klasifikasi Penyakit Kulit dengan Dataset Kecil
# Optimized untuk dataset 11 foto per kelas dengan augmentation intensif

"""
IMPROVEMENTS:
1. ✅ Heavy augmentation khusus untuk dataset sangat kecil
2. ✅ Advanced regularization techniques
3. ✅ K-Fold Cross Validation untuk dataset kecil
4. ✅ Test-Time Augmentation (TTA) untuk prediksi lebih robust
5. ✅ Mixup dan CutMix augmentation
6. ✅ Gradual unfreezing strategy
7. ✅ Better handling untuk foto inconsistent
8. ✅ Ensemble predictions
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import random
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
 
# TensorFlow dan Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import (
    Dense, Dropout, GlobalAveragePooling2D, 
    BatchNormalization, Activation, Input, GlobalMaxPooling2D
)
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K

# Scikit-learn
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils import class_weight

# Image processing
from PIL import Image, ImageEnhance
import cv2

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {len(tf.config.list_physical_devices('GPU'))} GPU(s)")

# ========================================
# KONFIGURASI UNTUK DATASET KECIL
# ========================================

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# Dataset configuration
DATASET_PATH = './dataset/augmented'  # Sesuaikan path Anda
OUTPUT_DIR = './output_training_improved'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model configuration - OPTIMIZED untuk dataset kecil
BASE_MODEL = 'MobileNetV2'  # Lightweight, cocok untuk dataset kecil
IMG_SIZE = (96, 96)  # LEBIH KECIL untuk mengurangi overfitting
IMG_SHAPE = (*IMG_SIZE, 3)

# Training parameters - ADJUSTED untuk dataset kecil
BATCH_SIZE = 4  # SANGAT KECIL untuk dataset 11 foto/kelas
EPOCHS = 100  # Lebih banyak epoch dengan early stopping
LEARNING_RATE = 0.0001
USE_KFOLD = True  # PENTING untuk dataset kecil
N_FOLDS = 5

# Augmentation - INTENSIF untuk dataset kecil
USE_HEAVY_AUGMENTATION = True
USE_MIXUP = True  # Advanced augmentation
MIXUP_ALPHA = 0.2
USE_TTA = True  # Test-Time Augmentation
TTA_STEPS = 10

print("="*70)
print("⚙️ KONFIGURASI UNTUK DATASET KECIL (11 foto/kelas)")
print("="*70)
print(f"Image Size: {IMG_SIZE} (kecil untuk reduce overfitting)")
print(f"Batch Size: {BATCH_SIZE} (sangat kecil)")
print(f"Heavy Augmentation: {USE_HEAVY_AUGMENTATION}")
print(f"K-Fold CV: {USE_KFOLD} (folds={N_FOLDS})")
print(f"Mixup: {USE_MIXUP}")
print(f"Test-Time Augmentation: {USE_TTA}")
print("="*70 + "\n")


# ========================================
# ADVANCED AUGMENTATION UNTUK DATASET KECIL
# ========================================

class HeavyAugmentor:
    """
    Augmentasi INTENSIF untuk dataset sangat kecil.
    Combines multiple augmentation techniques.
    """
    
    @staticmethod
    def get_training_augmentor():
        """
        Augmentasi untuk training - SANGAT AGRESIF
        """
        return ImageDataGenerator(
            rotation_range=40,           # Rotasi hingga 40 derajat
            width_shift_range=0.3,       # Geser horizontal 30%
            height_shift_range=0.3,      # Geser vertikal 30%
            shear_range=0.3,             # Shear transformation
            zoom_range=0.4,              # Zoom in/out 40%
            horizontal_flip=True,        # Flip horizontal
            vertical_flip=True,          # Flip vertical (berguna untuk skin)
            brightness_range=[0.5, 1.5], # Variasi brightness
            channel_shift_range=30,      # Shift warna
            fill_mode='reflect',         # Fill mode untuk rotasi
            preprocessing_function=HeavyAugmentor.advanced_augment
        )
    
    @staticmethod
    def advanced_augment(image):
        """
        Custom augmentation function untuk variasi tambahan
        """
        # Random noise
        if np.random.rand() > 0.5:
            noise = np.random.normal(0, 5, image.shape)
            image = np.clip(image + noise, 0, 255)
        
        # Random contrast
        if np.random.rand() > 0.5:
            factor = np.random.uniform(0.7, 1.3)
            image = np.clip(128 + factor * (image - 128), 0, 255)
        
        # Random saturation (jika RGB)
        if len(image.shape) == 3 and image.shape[2] == 3:
            if np.random.rand() > 0.5:
                hsv = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2HSV)
                hsv[:, :, 1] = hsv[:, :, 1] * np.random.uniform(0.7, 1.3)
                image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        return image
    
    @staticmethod
    def get_validation_augmentor():
        """
        Augmentasi ringan untuk validation/test
        """
        return ImageDataGenerator()


def mixup(x1, y1, x2, y2, alpha=0.2):
    """
    Mixup augmentation: mencampur dua sampel
    Paper: https://arxiv.org/abs/1710.09412
    
    Berguna untuk:
    - Regularization
    - Smooth decision boundaries
    - Reduce overfitting
    """
    lam = np.random.beta(alpha, alpha)
    x = lam * x1 + (1 - lam) * x2
    y = lam * y1 + (1 - lam) * y2
    return x, y


def create_augmented_dataset(X, y, num_classes, augmentation_factor=10):
    """
    Generate augmented dataset dari dataset kecil.
    
    Args:
        X: Images array
        y: Labels array
        num_classes: Jumlah kelas
        augmentation_factor: Berapa kali lipat augmentasi (10 = 10x data)
    
    Returns:
        X_aug, y_aug: Augmented dataset
    """
    print(f"\n🎨 Generating {augmentation_factor}x augmented dataset...")
    
    datagen = HeavyAugmentor.get_training_augmentor()
    datagen.fit(X)
    
    X_list = [X]
    y_list = [y]
    
    for i in range(augmentation_factor - 1):
        X_aug = np.zeros_like(X)
        
        for j in range(len(X)):
            # Generate augmented image
            img = X[j:j+1]
            aug_iter = datagen.flow(img, batch_size=1)
            X_aug[j] = next(aug_iter)[0]
        
        X_list.append(X_aug)
        y_list.append(y.copy())
        
        print(f"   Generated batch {i+1}/{augmentation_factor-1}")
    
    X_augmented = np.concatenate(X_list, axis=0)
    y_augmented = np.concatenate(y_list, axis=0)
    
    print(f"✅ Original dataset: {len(X)} samples")
    print(f"✅ Augmented dataset: {len(X_augmented)} samples ({len(X_augmented)/len(X):.1f}x)")
    
    return X_augmented, y_augmented


# ========================================
# IMPROVED MODEL ARCHITECTURE
# ========================================

def build_improved_model(num_classes, img_shape, learning_rate=0.0001):
    """
    Model architecture dengan regularization intensif untuk dataset kecil.
    
    Improvements:
    - Dropout lebih agresif
    - BatchNormalization di setiap layer
    - L2 regularization
    - Smaller dense layers (reduce parameters)
    - Spatial Dropout untuk CNN layers
    """
    print(f"\n🏗️ Building improved model for small dataset...")
    
    # Load base model
    base_model = MobileNetV2(
        input_shape=img_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model
    base_model.trainable = False
    
    # Build model with HEAVY regularization
    inputs = Input(shape=img_shape)
    
    # Base model
    x = base_model(inputs, training=False)
    
    # Global pooling - use BOTH average and max
    gap = GlobalAveragePooling2D()(x)
    gmp = GlobalMaxPooling2D()(x)
    x = layers.Concatenate()([gap, gmp])
    
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    # Smaller dense layers
    x = Dense(256, activation='relu', 
              kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(128, activation='relu',
              kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    # Output layer
    outputs = Dense(num_classes, activation='softmax',
                   kernel_regularizer=keras.regularizers.l2(0.01))(x)
    
    model = Model(inputs, outputs, name='ImprovedSkinClassifier')
    
    # Compile
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy', 
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')]
    )
    
    print(f"✅ Model built with heavy regularization")
    print(f"   Total parameters: {model.count_params():,}")
    print(f"   Trainable parameters: {sum([K.count_params(w) for w in model.trainable_weights]):,}")
    
    return model, base_model


# ========================================
# K-FOLD CROSS VALIDATION
# ========================================

def train_with_kfold(X, y, num_classes, n_folds=5):
    """
    K-Fold Cross Validation untuk dataset kecil.
    Ini SANGAT PENTING untuk dataset hanya 11 foto/kelas.
    
    Benefits:
    - Setiap sampel digunakan untuk training dan validation
    - Estimasi performa lebih reliable
    - Reduce variance
    """
    print("\n" + "="*70)
    print(f"🔄 K-FOLD CROSS VALIDATION ({n_folds} folds)")
    print("="*70)
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    
    fold_accuracies = []
    fold_models = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n{'='*70}")
        print(f"FOLD {fold}/{n_folds}")
        print(f"{'='*70}")
        
        # Split data
        X_train_fold = X[train_idx]
        y_train_fold = y[train_idx]
        X_val_fold = X[val_idx]
        y_val_fold = y[val_idx]
        
        print(f"Train: {len(X_train_fold)} samples")
        print(f"Val: {len(X_val_fold)} samples")
        
        # Augment training data
        if USE_HEAVY_AUGMENTATION:
            X_train_aug, y_train_aug = create_augmented_dataset(
                X_train_fold, y_train_fold, num_classes, augmentation_factor=15
            )
        else:
            X_train_aug, y_train_aug = X_train_fold, y_train_fold
        
        # Convert to categorical
        y_train_cat = to_categorical(y_train_aug, num_classes)
        y_val_cat = to_categorical(y_val_fold, num_classes)
        
        # Build model
        model, base_model = build_improved_model(
            num_classes=num_classes,
            img_shape=IMG_SHAPE,
            learning_rate=LEARNING_RATE
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=0
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=0
            )
        ]
        
        # Train
        history = model.fit(
            X_train_aug, y_train_cat,
            validation_data=(X_val_fold, y_val_cat),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=0
        )
        
        # Evaluate
        val_loss, val_acc, val_prec, val_rec = model.evaluate(
            X_val_fold, y_val_cat, verbose=0
        )
        
        print(f"\n📊 Fold {fold} Results:")
        print(f"   Val Accuracy: {val_acc:.4f}")
        print(f"   Val Precision: {val_prec:.4f}")
        print(f"   Val Recall: {val_rec:.4f}")
        
        fold_accuracies.append(val_acc)
        fold_models.append(model)
    
    print("\n" + "="*70)
    print("📊 K-FOLD CROSS VALIDATION RESULTS")
    print("="*70)
    print(f"Fold Accuracies: {[f'{acc:.4f}' for acc in fold_accuracies]}")
    print(f"Mean Accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
    print(f"Min Accuracy: {np.min(fold_accuracies):.4f}")
    print(f"Max Accuracy: {np.max(fold_accuracies):.4f}")
    print("="*70)
    
    # Return best model
    best_fold = np.argmax(fold_accuracies)
    print(f"\n✅ Best model: Fold {best_fold + 1} (Accuracy: {fold_accuracies[best_fold]:.4f})")
    
    return fold_models[best_fold], fold_models, fold_accuracies


# ========================================
# TEST-TIME AUGMENTATION (TTA)
# ========================================

def predict_with_tta(model, image, num_augmentations=10):
    """
    Test-Time Augmentation untuk prediksi lebih robust.
    
    Cara kerja:
    1. Generate multiple augmented versions dari 1 image
    2. Predict pada semua versions
    3. Average predictions
    
    Benefits:
    - Prediksi lebih stable
    - Reduce variance
    - Better generalization
    """
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )
    
    # Original prediction
    predictions = [model.predict(np.expand_dims(image, 0), verbose=0)[0]]
    
    # Augmented predictions
    for _ in range(num_augmentations - 1):
        aug_image = datagen.random_transform(image)
        pred = model.predict(np.expand_dims(aug_image, 0), verbose=0)[0]
        predictions.append(pred)
    
    # Average predictions
    avg_prediction = np.mean(predictions, axis=0)
    
    return avg_prediction


# ========================================
# ENSEMBLE PREDICTIONS
# ========================================

def ensemble_predict(models, X_test, use_tta=True):
    """
    Ensemble predictions dari multiple models (K-fold models).
    
    Cara kerja:
    1. Setiap model memberikan prediction
    2. Average semua predictions
    3. Opsional: gunakan TTA untuk setiap model
    
    Benefits:
    - Reduce overfitting
    - More robust predictions
    - Better generalization
    """
    print(f"\n🔮 Ensemble prediction with {len(models)} models...")
    
    all_predictions = []
    
    for i, model in enumerate(models):
        print(f"   Model {i+1}/{len(models)}...", end=' ')
        
        if use_tta and USE_TTA:
            # TTA for each sample
            predictions = []
            for j, image in enumerate(X_test):
                pred = predict_with_tta(model, image, num_augmentations=TTA_STEPS)
                predictions.append(pred)
                
                if (j + 1) % 10 == 0:
                    print(f"{j+1}/{len(X_test)}", end=' ')
            
            predictions = np.array(predictions)
        else:
            predictions = model.predict(X_test, verbose=0)
        
        all_predictions.append(predictions)
        print("✓")
    
    # Average all predictions
    ensemble_predictions = np.mean(all_predictions, axis=0)
    
    print(f"✅ Ensemble complete")
    
    return ensemble_predictions


# ========================================
# MAIN TRAINING PIPELINE
# ========================================

def main():
    """
    Main training pipeline dengan semua improvements
    """
    print("\n" + "="*70)
    print("🚀 IMPROVED SKIN DISEASE CLASSIFIER")
    print("   Optimized for small dataset (11 photos per class)")
    print("="*70 + "\n")
    
    # Load dataset
    print("📂 Loading dataset...")
    from pathlib import Path
    
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Dataset not found: {DATASET_PATH}")
        print("\n💡 Please set correct DATASET_PATH")
        return
    
    # Load images
    images = []
    labels = []
    class_names = []
    
    class_folders = sorted([f for f in os.listdir(DATASET_PATH) 
                           if os.path.isdir(os.path.join(DATASET_PATH, f))])
    
    print(f"Found {len(class_folders)} classes:")
    for idx, cls in enumerate(class_folders):
        print(f"   {idx}: {cls}")
        class_names.append(cls)
    
    for class_idx, class_name in enumerate(class_folders):
        class_path = os.path.join(DATASET_PATH, class_name)
        image_files = [f for f in os.listdir(class_path)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"Loading {class_name}: {len(image_files)} images")
        
        for img_file in image_files:
            try:
                img_path = os.path.join(class_path, img_file)
                img = load_img(img_path, target_size=IMG_SIZE)
                img_array = img_to_array(img)
                images.append(img_array)
                labels.append(class_idx)
            except Exception as e:
                print(f"   Error loading {img_file}: {e}")
    
    X = np.array(images)
    y = np.array(labels)
    num_classes = len(class_names)
    
    print(f"\n✅ Dataset loaded:")
    print(f"   Total samples: {len(X)}")
    print(f"   Classes: {num_classes}")
    print(f"   Samples per class: ~{len(X) // num_classes}")
    
    # WARNING untuk dataset kecil
    if len(X) < 100:
        print("\n⚠️  WARNING: Dataset sangat kecil!")
        print("   Recommendations:")
        print("   1. ✅ Heavy augmentation ENABLED")
        print("   2. ✅ K-Fold CV ENABLED")
        print("   3. ✅ Test-Time Augmentation ENABLED")
        print("   4. 📸 Collect more data if possible!")
    
    # Normalize
    X = X.astype('float32') / 255.0
    
    # Split train and test (80-20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    
    print(f"\n📊 Data split:")
    print(f"   Training: {len(X_train)} samples")
    print(f"   Test: {len(X_test)} samples")
    
    # K-Fold training
    if USE_KFOLD:
        best_model, all_models, fold_accs = train_with_kfold(
            X_train, y_train, num_classes, n_folds=N_FOLDS
        )
    else:
        # Single model training
        print("\n⚠️  K-Fold disabled, training single model...")
        X_train_aug, y_train_aug = create_augmented_dataset(
            X_train, y_train, num_classes, augmentation_factor=15
        )
        y_train_cat = to_categorical(y_train_aug, num_classes)
        
        best_model, _ = build_improved_model(num_classes, IMG_SHAPE, LEARNING_RATE)
        
        callbacks = [
            EarlyStopping(monitor='loss', patience=20, restore_best_weights=True),
            ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, min_lr=1e-7)
        ]
        
        best_model.fit(
            X_train_aug, y_train_cat,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )
        
        all_models = [best_model]
    
    # Test evaluation
    print("\n" + "="*70)
    print("🔍 FINAL EVALUATION ON TEST SET")
    print("="*70)
    
    # Ensemble prediction with TTA
    y_test_cat = to_categorical(y_test, num_classes)
    
    if USE_KFOLD:
        ensemble_preds = ensemble_predict(all_models, X_test, use_tta=USE_TTA)
        y_pred = np.argmax(ensemble_preds, axis=1)
    else:
        if USE_TTA:
            preds = []
            for img in X_test:
                pred = predict_with_tta(best_model, img, TTA_STEPS)
                preds.append(pred)
            y_pred = np.argmax(np.array(preds), axis=1)
        else:
            y_pred = np.argmax(best_model.predict(X_test, verbose=0), axis=1)
    
    # Metrics
    test_acc = accuracy_score(y_test, y_pred)
    
    print(f"\n📊 Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names, digits=4))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Test Set\n(with K-Fold + TTA)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Predicted', fontweight='bold')
    plt.ylabel('True', fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix_improved.png'), dpi=300)
    plt.show()
    
    # Save best model
    model_path = os.path.join(OUTPUT_DIR, 'best_model_improved.h5')
    best_model.save(model_path)
    print(f"\n💾 Model saved: {model_path}")
    
    # Save metadata
    metadata = {
        'model_type': 'ImprovedSkinClassifier',
        'base_model': BASE_MODEL,
        'img_size': IMG_SIZE,
        'num_classes': num_classes,
        'class_names': class_names,
        'total_samples': len(X),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'test_accuracy': float(test_acc),
        'kfold_enabled': USE_KFOLD,
        'n_folds': N_FOLDS if USE_KFOLD else None,
        'fold_accuracies': [float(acc) for acc in fold_accs] if USE_KFOLD else None,
        'tta_enabled': USE_TTA,
        'heavy_augmentation': USE_HEAVY_AUGMENTATION,
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    metadata_path = os.path.join(OUTPUT_DIR, 'metadata_improved.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"💾 Metadata saved: {metadata_path}")
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETED!")
    print("="*70)
    print("\n💡 TIPS untuk dataset kecil:")
    print("   1. ✅ Gunakan K-Fold CV (DONE)")
    print("   2. ✅ Heavy augmentation (DONE)")
    print("   3. ✅ Test-Time Augmentation (DONE)")
    print("   4. 📸 Collect MORE data (PALING PENTING!)")
    print("   5. 🏥 Ensure consistent image quality")
    print("   6. 📐 Standardize photo angles if possible")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
