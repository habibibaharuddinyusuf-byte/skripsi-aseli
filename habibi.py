"""
MODUL UTILITAS DETEKSI PENYAKIT TANAMAN
========================================
File: plant_disease_utils.py

Berisi semua class dan fungsi yang dibutuhkan untuk:
- Data loading & preprocessing
- Model building (CNN & Transfer Learning)
- Training & callbacks
- Visualisasi hasil
- Evaluasi model
- Prediksi gambar baru

Author: Plant Disease Detection System
Version: 1.0
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from pathlib import Path
import cv2
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# ================== KONFIGURASI DEFAULT ==================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001

# ================== 1. DATA LOADER ==================
class PlantDiseaseDataLoader:
    """
    Class untuk memuat dan memproses dataset tanaman
    
    Features:
    - Data augmentation untuk training set
    - Automatic class detection
    - Sample visualization
    - Class distribution analysis
    """
    
    def __init__(self, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
        """
        Initialize data loader
        
        Args:
            img_size (int): Ukuran gambar untuk resize
            batch_size (int): Ukuran batch untuk training
        """
        self.img_size = img_size
        self.batch_size = batch_size
        
    def create_data_generators(self, train_path, val_path, test_path):
        """
        Membuat data generator dengan augmentasi
        
        Args:
            train_path (str): Path ke folder training
            val_path (str): Path ke folder validation
            test_path (str): Path ke folder test
            
        Returns:
            tuple: (train_generator, val_generator, test_generator)
        """
        
        # Data augmentation untuk training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Hanya rescaling untuk validation dan test
        val_test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Load data
        train_generator = train_datagen.flow_from_directory(
            train_path,
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True
        )
        
        val_generator = val_test_datagen.flow_from_directory(
            val_path,
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        test_generator = val_test_datagen.flow_from_directory(
            test_path,
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        return train_generator, val_generator, test_generator
    
    def get_class_names(self, generator):
        """
        Mendapatkan nama kelas dari generator
        
        Args:
            generator: Data generator
            
        Returns:
            list: List nama kelas
        """
        return list(generator.class_indices.keys())
    
    def show_sample_images(self, generator, num_samples=9):
        """
        Menampilkan contoh gambar dari dataset
        
        Args:
            generator: Data generator
            num_samples (int): Jumlah gambar yang ditampilkan
        """
        x_batch, y_batch = next(generator)
        class_names = self.get_class_names(generator)
        
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        axes = axes.ravel()
        
        for i in range(min(num_samples, len(x_batch))):
            axes[i].imshow(x_batch[i])
            class_idx = np.argmax(y_batch[i])
            axes[i].set_title(class_names[class_idx], fontsize=10, fontweight='bold')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()

# ================== 2. MODEL BUILDER ==================
class PlantDiseaseModel:
    """
    Class untuk membuat dan melatih model CNN
    
    Supports:
    - Custom CNN dari scratch
    - Transfer Learning dengan MobileNetV2
    - Automatic callbacks (ModelCheckpoint, EarlyStopping, ReduceLROnPlateau)
    """
    
    def __init__(self, num_classes, img_size=IMG_SIZE):
        """
        Initialize model builder
        
        Args:
            num_classes (int): Jumlah kelas untuk klasifikasi
            img_size (int): Ukuran input gambar
        """
        self.num_classes = num_classes
        self.img_size = img_size
        self.model = None
        
    def build_cnn_model(self):
        """
        Membangun model CNN dari scratch
        
        Architecture:
        - 4 Convolutional blocks dengan BatchNorm dan Dropout
        - 2 Fully connected layers
        - Softmax output
        
        Returns:
            keras.Model: Model CNN
        """
        model = keras.Sequential([
            # Block 1
            layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                         input_shape=(self.img_size, self.img_size, 3)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 2
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 3
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 4
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fully Connected
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        return model
    
    def build_transfer_learning_model(self, base_model_name='MobileNetV2'):
        """
        Membangun model menggunakan Transfer Learning
        
        Args:
            base_model_name (str): Nama base model ('MobileNetV2', 'ResNet50', 'EfficientNetB0')
        
        Returns:
            keras.Model: Transfer learning model
        """
        # Load base model
        if base_model_name == 'MobileNetV2':
            base_model = keras.applications.MobileNetV2(
                input_shape=(self.img_size, self.img_size, 3),
                include_top=False,
                weights='imagenet'
            )
        elif base_model_name == 'ResNet50':
            base_model = keras.applications.ResNet50(
                input_shape=(self.img_size, self.img_size, 3),
                include_top=False,
                weights='imagenet'
            )
        elif base_model_name == 'EfficientNetB0':
            base_model = keras.applications.EfficientNetB0(
                input_shape=(self.img_size, self.img_size, 3),
                include_top=False,
                weights='imagenet'
            )
        else:
            raise ValueError(f"Unknown base model: {base_model_name}")
        
        # Freeze base model
        base_model.trainable = False
        
        # Build complete model
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        return model
    
    def unfreeze_base_model(self, num_layers_to_unfreeze=20):
        """
        Unfreeze beberapa layer dari base model untuk fine-tuning
        
        Args:
            num_layers_to_unfreeze (int): Jumlah layer yang akan di-unfreeze
        """
        if self.model is None:
            raise ValueError("Model belum dibuat. Panggil build_transfer_learning_model() terlebih dahulu.")
        
        base_model = self.model.layers[0]
        base_model.trainable = True
        
        # Freeze semua layer kecuali N layer terakhir
        for layer in base_model.layers[:-num_layers_to_unfreeze]:
            layer.trainable = False
        
        print(f"✅ Unfreezed {num_layers_to_unfreeze} layers dari base model")
    
    def compile_model(self, learning_rate=LEARNING_RATE):
        """
        Compile model dengan optimizer dan loss function
        
        Args:
            learning_rate (float): Learning rate untuk optimizer
        """
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
        )
        
    def train(self, train_gen, val_gen, epochs=EPOCHS, model_save_path='best_model.h5'):
        """
        Melatih model dengan callbacks
        
        Args:
            train_gen: Training data generator
            val_gen: Validation data generator
            epochs (int): Jumlah epoch
            model_save_path (str): Path untuk menyimpan model terbaik
            
        Returns:
            History: Training history
        """
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                model_save_path,
                save_best_only=True,
                monitor='val_accuracy',
                mode='max',
                verbose=1
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        history = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        return history

# ================== 3. VISUALIZER ==================
class Visualizer:
    """
    Class untuk visualisasi hasil training dan prediksi
    
    Features:
    - Training history plots
    - Confusion matrix
    - Sample predictions
    - Class distribution
    """
    
    @staticmethod
    def plot_training_history(history, save_path='training_history.png'):
        """
        Plot training history (accuracy & loss)
        
        Args:
            history: Training history dari model.fit()
            save_path (str): Path untuk menyimpan plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy
        axes[0].plot(history.history['accuracy'], label='Train Accuracy', marker='o', linewidth=2)
        axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', marker='s', linewidth=2)
        axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Accuracy', fontsize=12)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Loss
        axes[1].plot(history.history['loss'], label='Train Loss', marker='o', linewidth=2)
        axes[1].plot(history.history['val_loss'], label='Val Loss', marker='s', linewidth=2)
        axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Loss', fontsize=12)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        # Print best values
        best_train_acc = max(history.history['accuracy'])
        best_val_acc = max(history.history['val_accuracy'])
        best_epoch_acc = history.history['val_accuracy'].index(best_val_acc) + 1
        
        print(f"\n{'='*60}")
        print(f"📊 TRAINING SUMMARY")
        print(f"{'='*60}")
        print(f"Best Training Accuracy: {best_train_acc*100:.2f}%")
        print(f"Best Validation Accuracy: {best_val_acc*100:.2f}% (Epoch {best_epoch_acc})")
        print(f"{'='*60}")
    
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, class_names, save_path='confusion_matrix.png'):
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List nama kelas
            save_path (str): Path untuk menyimpan plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(max(12, len(class_names)//2), max(10, len(class_names)//2)))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names,
                   cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_sample_predictions(model, test_gen, class_names, num_samples=9, save_path='sample_predictions.png'):
        """
        Plot contoh prediksi dari test set
        
        Args:
            model: Trained model
            test_gen: Test data generator
            class_names: List nama kelas
            num_samples (int): Jumlah sample yang ditampilkan
            save_path (str): Path untuk menyimpan plot
        """
        x_batch, y_batch = next(test_gen)
        predictions = model.predict(x_batch[:num_samples], verbose=0)
        
        rows = int(np.ceil(num_samples / 3))
        fig, axes = plt.subplots(rows, 3, figsize=(15, rows * 5))
        axes = axes.ravel() if num_samples > 1 else [axes]
        
        for i in range(num_samples):
            axes[i].imshow(x_batch[i])
            
            true_label = class_names[np.argmax(y_batch[i])]
            pred_label = class_names[np.argmax(predictions[i])]
            confidence = np.max(predictions[i]) * 100
            
            color = 'green' if true_label == pred_label else 'red'
            title = f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.1f}%'
            axes[i].set_title(title, color=color, fontsize=10, fontweight='bold')
            axes[i].axis('off')
        
        # Hide extra subplots
        for i in range(num_samples, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_class_distribution(generator, title='Class Distribution'):
        """
        Plot distribusi kelas dalam dataset
        
        Args:
            generator: Data generator
            title (str): Judul plot
        """
        class_names = list(generator.class_indices.keys())
        class_counts = np.bincount(generator.classes)
        
        plt.figure(figsize=(max(12, len(class_names)//2), 6))
        bars = plt.bar(range(len(class_names)), class_counts, color='steelblue', alpha=0.8)
        plt.xlabel('Class', fontsize=12, fontweight='bold')
        plt.ylabel('Number of Images', fontsize=12, fontweight='bold')
        plt.title(title, fontsize=14, fontweight='bold', pad=15)
        plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right', fontsize=9)
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        # Print statistics
        total = sum(class_counts)
        print(f"\n{'='*60}")
        print(f"📊 CLASS DISTRIBUTION STATISTICS")
        print(f"{'='*60}")
        print(f"Total Images: {total}")
        print(f"Number of Classes: {len(class_names)}")
        print(f"Average per Class: {total/len(class_names):.1f}")
        print(f"Min: {min(class_counts)} | Max: {max(class_counts)}")
        print(f"{'='*60}")

# ================== 4. EVALUATOR ==================
class ModelEvaluator:
    """
    Class untuk evaluasi model
    
    Features:
    - Comprehensive metrics
    - Classification report
    - Per-class accuracy
    """
    
    @staticmethod
    def evaluate_model(model, test_gen, class_names):
        """
        Evaluasi lengkap model pada test set
        
        Args:
            model: Trained model
            test_gen: Test data generator
            class_names: List nama kelas
            
        Returns:
            tuple: (y_true, y_pred_classes, test_acc)
        """
        print("🔍 Evaluating model on test set...")
        print("="*60)
        
        # Get predictions
        y_pred = model.predict(test_gen, verbose=1)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = test_gen.classes
        
        # Calculate metrics
        results = model.evaluate(test_gen, verbose=0)
        test_loss = results[0]
        test_acc = results[1]
        test_top3 = results[2] if len(results) > 2 else None
        
        print(f"\n{'='*60}")
        print("📊 HASIL EVALUASI MODEL")
        print(f"{'='*60}")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc*100:.2f}%")
        if test_top3:
            print(f"Test Top-3 Accuracy: {test_top3*100:.2f}%")
        print(f"{'='*60}")
        
        # Classification report
        print(f"\n📋 CLASSIFICATION REPORT:")
        print("="*60)
        report = classification_report(y_true, y_pred_classes, 
                                      target_names=class_names,
                                      digits=4)
        print(report)
        print("="*60)
        
        return y_true, y_pred_classes, test_acc

# ================== 5. PREDICTOR ==================
class PlantDiseasePredictor:
    """
    Class untuk prediksi penyakit tanaman dari gambar baru
    
    Features:
    - Single image prediction
    - Batch prediction
    - Visualization support
    - Top-K predictions
    """
    
    def __init__(self, model_path, class_names, img_size=IMG_SIZE):
        """
        Initialize predictor
        
        Args:
            model_path (str): Path ke model file (.h5)
            class_names (list): List nama kelas
            img_size (int): Ukuran input gambar
        """
        print(f"⏳ Loading model from {model_path}...")
        self.model = keras.models.load_model(model_path)
        self.class_names = class_names
        self.img_size = img_size
        print("✅ Model loaded successfully!")
    
    def preprocess_image(self, img_path):
        """
        Preprocessing gambar untuk prediksi
        
        Args:
            img_path (str): Path ke gambar
            
        Returns:
            tuple: (preprocessed_image, original_image)
        """
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Cannot read image from {img_path}")
        
        # Convert BGR to RGB
        original_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize and normalize
        img_resized = cv2.resize(original_img, (self.img_size, self.img_size))
        img_normalized = img_resized / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        return img_batch, original_img
    
    def predict(self, img_path, top_k=3):
        """
        Prediksi penyakit dari gambar
        
        Args:
            img_path (str): Path ke gambar
            top_k (int): Jumlah prediksi teratas yang dikembalikan
            
        Returns:
            list: List dictionary dengan 'class' dan 'confidence'
        """
        img, _ = self.preprocess_image(img_path)
        predictions = self.model.predict(img, verbose=0)[0]
        
        # Get top k predictions
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'class': self.class_names[idx],
                'confidence': float(predictions[idx] * 100)
            })
        
        return results
    
    def predict_and_visualize(self, img_path, save_path='prediction_result.png'):
        """
        Prediksi dan visualisasi hasil
        
        Args:
            img_path (str): Path ke gambar
            save_path (str): Path untuk menyimpan hasil visualisasi
            
        Returns:
            list: Hasil prediksi
        """
        results = self.predict(img_path)
        
        # Load original image
        _, original_img = self.preprocess_image(img_path)
        
        # Create figure
        fig = plt.figure(figsize=(14, 6))
        
        # Show image
        ax1 = plt.subplot(1, 2, 1)
        ax1.imshow(original_img)
        ax1.set_title('Input Image', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Show predictions
        ax2 = plt.subplot(1, 2, 2)
        classes = [r['class'] for r in results]
        confidences = [r['confidence'] for r in results]
        
        # Color coding
        colors = []
        for i in range(len(classes)):
            if i == 0:
                colors.append('#2ecc71')  # Green for top prediction
            elif i == 1:
                colors.append('#f39c12')  # Orange for 2nd
            else:
                colors.append('#e74c3c')  # Red for others
        
        bars = ax2.barh(classes, confidences, color=colors, alpha=0.8)
        ax2.set_xlabel('Confidence (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Top Predictions', fontsize=14, fontweight='bold')
        ax2.set_xlim(0, 100)
        ax2.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Add confidence values on bars
        for i, (cls, conf) in enumerate(zip(classes, confidences)):
            ax2.text(conf + 2, i, f'{conf:.1f}%', va='center', fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        # Print results
        print(f"\n{'='*60}")
        print("🔬 HASIL PREDIKSI")
        print(f"{'='*60}")
        print(f"Image: {os.path.basename(img_path)}")
        print(f"{'='*60}")
        for i, result in enumerate(results, 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
            print(f"{emoji} {i}. {result['class']}: {result['confidence']:.2f}%")
        print(f"{'='*60}")
        
        return results
    
    def predict_batch(self, image_paths, visualize=True):
        """
        Prediksi batch gambar
        
        Args:
            image_paths (list): List path gambar
            visualize (bool): Tampilkan visualisasi atau tidak
            
        Returns:
            list: List hasil prediksi untuk setiap gambar
        """
        all_results = []
        
        print(f"\n🚀 Processing {len(image_paths)} images...")
        print("="*60)
        
        for i, img_path in enumerate(image_paths, 1):
            print(f"\n[{i}/{len(image_paths)}] Processing: {os.path.basename(img_path)}")
            try:
                if visualize:
                    results = self.predict_and_visualize(
                        img_path, 
                        save_path=f'prediction_{i}.png'
                    )
                else:
                    results = self.predict(img_path)
                    print(f"   Top prediction: {results[0]['class']} ({results[0]['confidence']:.2f}%)")
                
                all_results.append({
                    'image': img_path,
                    'predictions': results,
                    'top_class': results[0]['class'],
                    'top_confidence': results[0]['confidence']
                })
            except Exception as e:
                print(f"   ❌ Error: {e}")
                all_results.append({
                    'image': img_path,
                    'predictions': None,
                    'error': str(e)
                })
        
        print("\n" + "="*60)
        print("✅ Batch prediction completed!")
        print("="*60)
        
        return all_results

# ================== HELPER FUNCTIONS ==================
def save_class_names(class_names, filepath='class_names.txt'):
    """
    Simpan nama kelas ke file
    
    Args:
        class_names (list): List nama kelas
        filepath (str): Path file output
    """
    with open(filepath, 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")
    print(f"✅ Class names saved to {filepath}")

def load_class_names(filepath='class_names.txt'):
    """
    Load nama kelas dari file
    
    Args:
        filepath (str): Path file class names
        
    Returns:
        list: List nama kelas
    """
    with open(filepath, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    print(f"✅ Loaded {len(class_names)} classes from {filepath}")
    return class_names

def print_model_info(model):
    """
    Print informasi model
    
    Args:
        model: Keras model
    """
    print("\n" + "="*60)
    print("📋 MODEL INFORMATION")
    print("="*60)
    model.summary()
    print("="*60)
    
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    
    print(f"\n📊 PARAMETER SUMMARY")
    print(f"{'='*60}")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Non-trainable Parameters: {non_trainable_params:,}")
    print(f"{'='*60}\n")

def create_sample_dataset_structure(base_path='sample_dataset'):
    """
    Create sample dataset structure (untuk testing)
    
    Args:
        base_path (str): Base path untuk dataset
    """
    folders = [
        os.path.join(base_path, 'train'),
        os.path.join(base_path, 'val'),
        os.path.join(base_path, 'test')
    ]
    
    sample_classes = ['healthy', 'leaf_spot', 'powdery_mildew', 'rust']
    
    for folder in folders:
        for class_name in sample_classes:
            class_path = os.path.join(folder, class_name)
            os.makedirs(class_path, exist_ok=True)
    
    print(f"✅ Sample dataset structure created at: {base_path}")
    print(f"   Classes: {', '.join(sample_classes)}")
    print(f"   ⚠️ Please add your images to these folders!")

def get_gpu_info():
    """
    Get GPU information
    
    Returns:
        dict: GPU information
    """
    gpus = tf.config.list_physical_devices('GPU')
    
    info = {
        'gpu_available': len(gpus) > 0,
        'num_gpus': len(gpus),
        'gpu_names': []
    }
    
    if gpus:
        for gpu in gpus:
            info['gpu_names'].append(gpu.name)
            # Set memory growth
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except:
                pass
    
    return info

def print_gpu_info():
    """Print GPU information"""
    info = get_gpu_info()
    
    print("\n" + "="*60)
    print("🖥️  GPU INFORMATION")
    print("="*60)
    
    if info['gpu_available']:
        print(f"✅ GPU Available: Yes")
        print(f"📊 Number of GPUs: {info['num_gpus']}")
        for i, name in enumerate(info['gpu_names'], 1):
            print(f"   GPU {i}: {name}")
    else:
        print("❌ GPU Available: No")
        print("⚠️  Training will use CPU (slower)")
    
    print("="*60)

def calculate_steps(generator, batch_size):
    """
    Calculate steps per epoch
    
    Args:
        generator: Data generator
        batch_size (int): Batch size
        
    Returns:
        int: Steps per epoch
    """
    return int(np.ceil(generator.samples / batch_size))

def plot_model_comparison(histories, labels, save_path='model_comparison.png'):
    """
    Plot comparison of multiple training histories
    
    Args:
        histories (list): List of training histories
        labels (list): List of model labels
        save_path (str): Path to save plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    
    # Accuracy comparison
    for i, (history, label) in enumerate(zip(histories, labels)):
        color = colors[i % len(colors)]
        axes[0].plot(history.history['val_accuracy'], 
                    label=label, color=color, marker='o', linewidth=2)
    
    axes[0].set_title('Validation Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss comparison
    for i, (history, label) in enumerate(zip(histories, labels)):
        color = colors[i % len(colors)]
        axes[1].plot(history.history['val_loss'], 
                    label=label, color=color, marker='s', linewidth=2)
    
    axes[1].set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def export_model_to_tflite(model_path, output_path='model.tflite'):
    """
    Convert Keras model to TensorFlow Lite format
    
    Args:
        model_path (str): Path to Keras model (.h5)
        output_path (str): Output path for TFLite model
    """
    print(f"🔄 Converting model to TensorFlow Lite...")
    
    # Load model
    model = keras.models.load_model(model_path)
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    # Save
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    # Get file sizes
    original_size = os.path.getsize(model_path) / (1024*1024)
    tflite_size = os.path.getsize(output_path) / (1024*1024)
    
    print(f"✅ TFLite model saved to: {output_path}")
    print(f"📊 Original size: {original_size:.2f} MB")
    print(f"📊 TFLite size: {tflite_size:.2f} MB")
    print(f"📊 Compression: {(1 - tflite_size/original_size)*100:.1f}%")

def save_training_config(config, filepath='training_config.json'):
    """
    Save training configuration
    
    Args:
        config (dict): Configuration dictionary
        filepath (str): Output file path
    """
    import json
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"✅ Training config saved to {filepath}")

def load_training_config(filepath='training_config.json'):
    """
    Load training configuration
    
    Args:
        filepath (str): Config file path
        
    Returns:
        dict: Configuration dictionary
    """
    import json
    with open(filepath, 'r') as f:
        config = json.load(f)
    print(f"✅ Training config loaded from {filepath}")
    return config

# ================== ADVANCED FEATURES ==================
class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM)
    Untuk visualisasi area yang menjadi fokus model saat prediksi
    """
    
    def __init__(self, model, layer_name=None):
        """
        Initialize Grad-CAM
        
        Args:
            model: Keras model
            layer_name (str): Nama layer untuk visualisasi (default: last conv layer)
        """
        self.model = model
        
        # Find last conv layer if not specified
        if layer_name is None:
            for layer in reversed(model.layers):
                if 'conv' in layer.name.lower():
                    layer_name = layer.name
                    break
        
        self.layer_name = layer_name
        print(f"✅ Grad-CAM initialized with layer: {layer_name}")
    
    def generate_heatmap(self, img_array, class_idx):
        """
        Generate Grad-CAM heatmap
        
        Args:
            img_array: Preprocessed image array
            class_idx: Target class index
            
        Returns:
            numpy.ndarray: Heatmap
        """
        # Create gradient model
        grad_model = keras.models.Model(
            [self.model.inputs],
            [self.model.get_layer(self.layer_name).output, self.model.output]
        )
        
        # Compute gradient
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, class_idx]
        
        # Get gradients
        grads = tape.gradient(loss, conv_outputs)
        
        # Compute weights
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the channels
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
        return heatmap.numpy()
    
    def visualize(self, img_path, predictor, save_path='gradcam_result.png'):
        """
        Visualize Grad-CAM
        
        Args:
            img_path (str): Path to image
            predictor: PlantDiseasePredictor instance
            save_path (str): Path to save visualization
        """
        # Preprocess image
        img_array, original_img = predictor.preprocess_image(img_path)
        
        # Get prediction
        predictions = self.model.predict(img_array, verbose=0)[0]
        class_idx = np.argmax(predictions)
        class_name = predictor.class_names[class_idx]
        confidence = predictions[class_idx] * 100
        
        # Generate heatmap
        heatmap = self.generate_heatmap(img_array, class_idx)
        
        # Resize heatmap to image size
        heatmap_resized = cv2.resize(heatmap, 
                                     (original_img.shape[1], original_img.shape[0]))
        
        # Convert to RGB
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized), 
            cv2.COLORMAP_JET
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Superimpose
        superimposed = cv2.addWeighted(original_img, 0.6, heatmap_colored, 0.4, 0)
        
        # Visualize
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        axes[0].imshow(original_img)
        axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(heatmap_resized, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        axes[2].imshow(superimposed)
        axes[2].set_title(f'Prediction: {class_name}\nConfidence: {confidence:.1f}%',
                         fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Grad-CAM visualization saved to {save_path}")

# ================== MAIN EXECUTION GUARD ==================
if __name__ == "__main__":
    print("="*60)
    print("🌱 PLANT DISEASE DETECTION UTILS")
    print("="*60)
    print("This is a utility module. Import it in your notebook:")
    print("\n  from plant_disease_utils import *")
    print("\nOr import specific components:")
    print("\n  from plant_disease_utils import (")
    print("      PlantDiseaseDataLoader,")
    print("      PlantDiseaseModel,")
    print("      Visualizer,")
    print("      ModelEvaluator,")
    print("      PlantDiseasePredictor")
    print("  )")
    print("="*60)
    
    # Print GPU info
    print_gpu_info()
    
    print("\n✨ Module loaded successfully!")
    print("📖 Check the notebook for complete usage examples.")
    print("="*60)
    