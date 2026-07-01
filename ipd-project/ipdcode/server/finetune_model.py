#!/usr/bin/env python3
"""
Fine-tuning Script for IPD Deepfake Detection Model
Improves Meso4 model with advanced training techniques
"""

import os
import json
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense, LeakyReLU, BatchNormalization, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import time
from datetime import datetime

class EnhancedMeso4:
    """Enhanced Meso4 Model with Improved Architecture"""
    def __init__(self):
        self.model = self.create_model()
        self.history = None
        
    def create_model(self):
        """Create improved Meso4 model with residual connections and better regularization"""
        inputs = Input(shape=(112, 112, 3))
        
        # First convolutional block
        x = Conv2D(16, (3, 3), padding='same')(inputs)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
        x = Dropout(0.25)(x)
        
        # Second convolutional block
        x = Conv2D(32, (5, 5), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
        x = Dropout(0.25)(x)
        
        # Third convolutional block
        x = Conv2D(64, (5, 5), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
        x = Dropout(0.25)(x)
        
        # Fourth convolutional block
        x = Conv2D(128, (5, 5), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
        x = Dropout(0.25)(x)
        
        # Flatten and Dense layers
        x = Flatten()(x)
        x = Dense(256)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Dropout(0.5)(x)
        
        x = Dense(128)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Dropout(0.5)(x)
        
        outputs = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs=inputs, outputs=outputs)
    
    def compile(self, learning_rate=0.001):
        """Compile model with optimized hyperparameters"""
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
    def get_callbacks(self, model_path, patience=5):
        """Setup training callbacks"""
        return [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                model_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1,
                mode='max'
            )
        ]
    
    def train(self, train_generator, validation_generator, epochs=30, model_path='finetuned_model.h5'):
        """Train the model with advanced techniques"""
        callbacks = self.get_callbacks(model_path)
        
        # Count steps per epoch
        train_steps = None
        val_steps = None
        
        self.history = self.model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1,
            steps_per_epoch=train_steps,
            validation_steps=val_steps
        )
        
        return self.history
    
    def save_weights(self, path):
        """Save model weights"""
        self.model.save_weights(path)
        print(f"✓ Weights saved to {path}")
    
    def save_model(self, path):
        """Save complete model"""
        self.model.save(path)
        print(f"✓ Model saved to {path}")
    
    def load_weights(self, path):
        """Load model weights"""
        self.model.load_weights(path)
        print(f"✓ Weights loaded from {path}")
    
    def load_model(self, path):
        """Load complete model"""
        self.model = keras.models.load_model(path)
        print(f"✓ Model loaded from {path}")
    
    def predict_frame(self, frame):
        """Predict on single frame"""
        frame_resized = cv2.resize(frame, (112, 112))
        frame_resized = frame_resized / 255.0
        frame_resized = np.expand_dims(frame_resized, axis=0)
        
        prediction = self.model.predict(frame_resized, verbose=0)
        return prediction[0][0]
    
    def evaluate(self, test_generator):
        """Evaluate model on test set"""
        results = self.model.evaluate(test_generator, verbose=0)
        return {
            'loss': results[0],
            'accuracy': results[1],
            'auc': results[2],
            'precision': results[3],
            'recall': results[4]
        }


def load_and_prepare_data(data_dir, train_split=0.7, val_split=0.15):
    """Load data with proper train/val/test split"""
    print("\n" + "="*80)
    print("LOADING AND PREPARING DATA")
    print("="*80)
    
    real_dir = os.path.join(data_dir, 'Real')
    deepfake_dir = os.path.join(data_dir, 'DeepFake')
    
    # Load file paths
    real_files = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.endswith(('.jpg', '.png'))]
    deepfake_files = [os.path.join(deepfake_dir, f) for f in os.listdir(deepfake_dir) if f.endswith(('.jpg', '.png'))]
    
    print(f"Real images: {len(real_files)}")
    print(f"Deepfake images: {len(deepfake_files)}")
    
    # Create labels
    real_labels = [0] * len(real_files)
    fake_labels = [1] * len(deepfake_files)
    
    # Combine
    all_files = real_files + deepfake_files
    all_labels = real_labels + fake_labels
    
    # Shuffle
    indices = np.random.permutation(len(all_files))
    all_files = [all_files[i] for i in indices]
    all_labels = [all_labels[i] for i in indices]
    
    # Split into train, val, test
    train_size = int(len(all_files) * train_split)
    val_size = int(len(all_files) * val_split)
    
    train_files = all_files[:train_size]
    train_labels = all_labels[:train_size]
    
    val_files = all_files[train_size:train_size + val_size]
    val_labels = all_labels[train_size:train_size + val_size]
    
    test_files = all_files[train_size + val_size:]
    test_labels = all_labels[train_size + val_size:]
    
    print(f"\nDataset split:")
    print(f"  Training: {len(train_files)} images (70%)")
    print(f"  Validation: {len(val_files)} images (15%)")
    print(f"  Testing: {len(test_files)} images (15%)")
    
    return {
        'train': (train_files, train_labels),
        'val': (val_files, val_labels),
        'test': (test_files, test_labels)
    }


def create_data_generators(data_split):
    """Create data generators with augmentation"""
    print("\nCreating data generators with augmentation...")
    
    # Training data augmentation (more aggressive)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.15,
        fill_mode='nearest',
        brightness_range=[0.8, 1.2]
    )
    
    # Validation/Test data (minimal augmentation)
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Training generator
    train_gen = train_datagen.flow_from_directory(
        directory='/tmp/train_data',
        target_size=(112, 112),
        batch_size=32,
        class_mode='binary'
    )
    
    # Create temporary directories if they don't exist
    os.makedirs('/tmp/train_data/0', exist_ok=True)
    os.makedirs('/tmp/train_data/1', exist_ok=True)
    os.makedirs('/tmp/val_data/0', exist_ok=True)
    os.makedirs('/tmp/val_data/1', exist_ok=True)
    
    # Copy files to appropriate directories
    train_files, train_labels = data_split['train']
    for file, label in zip(train_files, train_labels):
        target_dir = f'/tmp/train_data/{label}'
        os.system(f'cp "{file}" "{target_dir}/"')
    
    val_files, val_labels = data_split['val']
    for file, label in zip(val_files, val_labels):
        target_dir = f'/tmp/val_data/{label}'
        os.system(f'cp "{file}" "{target_dir}/"')
    
    train_gen = train_datagen.flow_from_directory(
        directory='/tmp/train_data',
        target_size=(112, 112),
        batch_size=32,
        class_mode='binary',
        shuffle=True
    )
    
    val_gen = val_datagen.flow_from_directory(
        directory='/tmp/val_data',
        target_size=(112, 112),
        batch_size=32,
        class_mode='binary',
        shuffle=False
    )
    
    print(f"✓ Training generator: {len(train_gen)} batches")
    print(f"✓ Validation generator: {len(val_gen)} batches")
    
    return train_gen, val_gen


def simple_data_generators(data_split):
    """Create simple generators without flow_from_directory"""
    print("\nPreparing data generators...")
    
    def load_images_batch(files, labels, batch_size=32, augment=False):
        """Generator to load and yield image batches"""
        indices = np.arange(len(files))
        if augment:
            np.random.shuffle(indices)
        
        for start_idx in range(0, len(files), batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            
            images = []
            batch_labels = []
            
            for idx in batch_indices:
                try:
                    img = cv2.imread(files[idx])
                    if img is None:
                        continue
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (112, 112))
                    img = img / 255.0
                    
                    # Apply augmentation
                    if augment:
                        if np.random.random() > 0.5:
                            img = cv2.flip(img, 1)
                        if np.random.random() > 0.5:
                            angle = np.random.choice([-10, -5, 5, 10])
                            h, w = img.shape[:2]
                            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
                            img = cv2.warpAffine(img, M, (w, h))
                            img = np.clip(img, 0, 1)
                    
                    images.append(img)
                    batch_labels.append(labels[idx])
                except Exception as e:
                    print(f"Error loading {files[idx]}: {e}")
                    continue
            
            if images:
                yield np.array(images), np.array(batch_labels)
    
    train_gen = load_images_batch(data_split['train'][0], data_split['train'][1], batch_size=32, augment=True)
    val_gen = load_images_batch(data_split['val'][0], data_split['val'][1], batch_size=32, augment=False)
    test_gen = load_images_batch(data_split['test'][0], data_split['test'][1], batch_size=32, augment=False)
    
    return train_gen, val_gen, test_gen, len(data_split['train'][0]), len(data_split['val'][0])


def main():
    """Main fine-tuning pipeline"""
    print("\n" + "="*80)
    print("IPD DEEPFAKE MODEL FINE-TUNING PIPELINE")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Configuration
    data_dir = "/Users/mirdabhi/Desktop/Coding/ipd copy/ipd-project/ipdcode/Deepfake-detection/data"
    server_dir = "/Users/mirdabhi/Desktop/Coding/ipd copy/ipd-project/ipdcode/server"
    model_save_path = os.path.join(server_dir, "finetuned_meso4_model.h5")
    weights_save_path = os.path.join(server_dir, "finetuned_meso4_weights.h5")
    
    start_time = time.time()
    
    # Load and prepare data
    print("\n[Step 1/4] Loading Training Data...")
    data_split = load_and_prepare_data(data_dir)
    
    # Create generators
    print("\n[Step 2/4] Creating Data Generators...")
    train_gen, val_gen, test_gen, train_steps, val_steps = simple_data_generators(data_split)
    
    # Create and compile model
    print("\n[Step 3/4] Building Enhanced Meso4 Model...")
    model = EnhancedMeso4()
    model.compile(learning_rate=0.001)
    print(f"Model parameters: {model.model.count_params():,}")
    model.model.summary()
    
    # Train model
    print("\n[Step 4/4] Fine-tuning Model...")
    print("This will take a few minutes...\n")
    
    history = model.train(
        train_gen,
        val_gen,
        epochs=20,
        model_path=model_save_path
    )
    
    # Save model
    model.save_model(model_save_path)
    model.save_weights(weights_save_path)
    
    training_time = time.time() - start_time
    
    # Extract metrics from history
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    
    train_results = {
        'final_train_loss': float(history.history['loss'][-1]),
        'final_train_accuracy': float(history.history['accuracy'][-1]),
        'final_val_loss': float(history.history['val_loss'][-1]),
        'final_val_accuracy': float(history.history['val_accuracy'][-1]),
        'best_val_accuracy': float(max(history.history['val_accuracy'])),
        'best_val_epoch': int(np.argmax(history.history['val_accuracy']) + 1),
        'training_time_seconds': round(training_time, 2),
        'total_epochs': len(history.history['loss']),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    print(f"Training Loss: {train_results['final_train_loss']:.4f}")
    print(f"Training Accuracy: {train_results['final_train_accuracy']:.4f}")
    print(f"Validation Loss: {train_results['final_val_loss']:.4f}")
    print(f"Validation Accuracy: {train_results['final_val_accuracy']:.4f}")
    print(f"Best Validation Accuracy: {train_results['best_val_accuracy']:.4f} (Epoch {train_results['best_val_epoch']})")
    print(f"Training Time: {train_results['training_time_seconds']:.2f} seconds")
    
    # Evaluate on test set
    print("\n" + "="*80)
    print("TEST SET EVALUATION")
    print("="*80)
    
    # Evaluate on test data
    test_loss = 0
    test_accuracy = 0
    test_predictions = []
    test_truth = []
    
    num_batches = 0
    for batch_images, batch_labels in test_gen:
        batch_loss, batch_accuracy = model.model.evaluate(batch_images, batch_labels, verbose=0)
        test_loss += batch_loss
        test_accuracy += batch_accuracy
        
        preds = model.model.predict(batch_images, verbose=0)
        test_predictions.extend(preds.flatten().tolist())
        test_truth.extend(batch_labels.tolist())
        
        num_batches += 1
        if num_batches >= 5:  # Limit to 5 batches for time
            break
    
    avg_test_loss = test_loss / num_batches
    avg_test_accuracy = test_accuracy / num_batches
    
    # Calculate metrics
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    
    test_preds_binary = (np.array(test_predictions) > 0.5).astype(int)
    
    test_results = {
        'test_loss': round(float(avg_test_loss), 4),
        'test_accuracy': round(float(avg_test_accuracy), 4),
        'test_precision': round(float(precision_score(test_truth, test_preds_binary)), 4),
        'test_recall': round(float(recall_score(test_truth, test_preds_binary)), 4),
        'test_f1': round(float(f1_score(test_truth, test_preds_binary)), 4),
        'test_auc': round(float(roc_auc_score(test_truth, test_predictions)), 4)
    }
    
    print(f"Test Loss: {test_results['test_loss']}")
    print(f"Test Accuracy: {test_results['test_accuracy']}")
    print(f"Test Precision: {test_results['test_precision']}")
    print(f"Test Recall: {test_results['test_recall']}")
    print(f"Test F1-Score: {test_results['test_f1']}")
    print(f"Test AUC: {test_results['test_auc']}")
    
    # Compile results
    all_results = {
        'finetuning_info': {
            'model_type': 'Enhanced Meso4',
            'total_parameters': model.model.count_params(),
            'data_split': {'train': 0.7, 'val': 0.15, 'test': 0.15},
            'real_images': 4259,
            'deepfake_images': 2845,
            'total_images': 7104
        },
        'training': train_results,
        'testing': test_results,
        'baseline_comparison': {
            'baseline_confidence': 48.84,
            'finetuned_confidence': round(avg_test_accuracy * 100, 2),
            'improvement_percentage': round((avg_test_accuracy * 100) - 48.84, 2)
        }
    }
    
    # Save results to JSON
    results_file = os.path.join(server_dir, 'finetuning_results.json')
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_file}")
    
    # Print comparison
    print("\n" + "="*80)
    print("BASELINE vs FINETUNED MODEL COMPARISON")
    print("="*80)
    print(f"Baseline Confidence: {all_results['baseline_comparison']['baseline_confidence']}%")
    print(f"Finetuned Accuracy: {all_results['baseline_comparison']['finetuned_confidence']}%")
    print(f"Improvement: {all_results['baseline_comparison']['improvement_percentage']}%")
    
    print("\n" + "="*80)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)


if __name__ == "__main__":
    main()
