import os
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pytz
import random

from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# Konfigurasi TensorFlow untuk CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.set_visible_devices([], 'GPU')
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)

print("=" * 70)
print("BISINDO TRAINING - MEDIUM DATASET (FAST TRAINING)")
print("=" * 70)

# Setup waktu
timeJKT = pytz.timezone('Asia/Jakarta') 
start_time = datetime.now(timeJKT).replace(microsecond=0)
print(f"Start Time: {start_time}\n")

# ============================================================================
# KONFIGURASI
# ============================================================================
DATA_DIR = "./Data"
SUBSAMPLE_DIR = "./Data_Medium"
IMAGES_PER_CLASS = 20000  # 20k per class = 520k total (good balance!)
IMG_SIZE = 64
BATCH_SIZE = 128
EPOCHS = 40  # Reduced from 50
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2

print(f"Configuration:")
print(f"  - Images per class: {IMAGES_PER_CLASS:,}")
print(f"  - Image size: {IMG_SIZE}x{IMG_SIZE}")
print(f"  - Batch size: {BATCH_SIZE}")
print(f"  - Epochs: {EPOCHS}")
print(f"  - Expected total images: ~{IMAGES_PER_CLASS * 26:,}")
print(f"  - Estimated training time: 45-60 minutes\n")

# ============================================================================
# CREATE SUBSAMPLED DATASET
# ============================================================================
def create_subsample_dataset(source_dir, dest_dir, images_per_class):
    """
    Create subsampled dataset with specified images per class
    """
    print(f"Creating subsampled dataset in '{dest_dir}'...")
    
    # Remove old subsample if exists
    if os.path.exists(dest_dir):
        print(f"  Removing old subsample directory...")
        shutil.rmtree(dest_dir)
    
    os.makedirs(dest_dir)
    
    class_folders = sorted([f for f in os.listdir(source_dir) 
                           if os.path.isdir(os.path.join(source_dir, f))])
    
    total_copied = 0
    
    for class_name in class_folders:
        source_class_dir = os.path.join(source_dir, class_name)
        dest_class_dir = os.path.join(dest_dir, class_name)
        os.makedirs(dest_class_dir)
        
        # Get all images
        all_images = [f for f in os.listdir(source_class_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Random sample
        num_to_copy = min(images_per_class, len(all_images))
        selected_images = random.sample(all_images, num_to_copy)
        
        # Copy images
        for img_name in selected_images:
            src = os.path.join(source_class_dir, img_name)
            dst = os.path.join(dest_class_dir, img_name)
            shutil.copy2(src, dst)
        
        total_copied += num_to_copy
        print(f"  ✓ {class_name}: copied {num_to_copy:,} images")
    
    print(f"\n✓ Subsample dataset created: {total_copied:,} images total\n")
    return dest_dir

# Check if subsample already exists
if os.path.exists(SUBSAMPLE_DIR):
    existing_count = sum([len(os.listdir(os.path.join(SUBSAMPLE_DIR, f))) 
                         for f in os.listdir(SUBSAMPLE_DIR) 
                         if os.path.isdir(os.path.join(SUBSAMPLE_DIR, f))])
    
    print(f"Found existing subsample dataset with {existing_count:,} images")
    use_existing = input("Use existing subsample? (yes/no): ").strip().lower()
    
    if use_existing != 'yes':
        DATA_DIR = create_subsample_dataset(DATA_DIR, SUBSAMPLE_DIR, IMAGES_PER_CLASS)
    else:
        DATA_DIR = SUBSAMPLE_DIR
        print("Using existing subsample dataset\n")
else:
    DATA_DIR = create_subsample_dataset(DATA_DIR, SUBSAMPLE_DIR, IMAGES_PER_CLASS)

# ============================================================================
# DATA GENERATORS
# ============================================================================
print("Setting up data generators...")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=VALIDATION_SPLIT
)

train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42
)

val_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    seed=42
)

num_classes = len(train_generator.class_indices)
class_names = list(train_generator.class_indices.keys())

print(f"✓ Training samples: {train_generator.samples:,}")
print(f"✓ Validation samples: {val_generator.samples:,}")
print(f"✓ Number of classes: {num_classes}")
print(f"✓ Steps per epoch: {train_generator.samples // BATCH_SIZE}")
print(f"✓ Validation steps: {val_generator.samples // BATCH_SIZE}\n")

# ============================================================================
# BUILD CNN MODEL
# ============================================================================
print("Building CNN model...")

model = Sequential([
    # Block 1
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    # Block 2
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    # Block 3
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    # Fully Connected
    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nModel Summary:")
model.summary()

# ============================================================================
# CALLBACKS
# ============================================================================
callbacks = [
    ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=8,  # Reduced patience
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=4,  # Reduced patience
        min_lr=1e-7,
        verbose=1
    )
]

# ============================================================================
# TRAINING
# ============================================================================
print("\n" + "=" * 70)
print("STARTING TRAINING...")
print("=" * 70)
print("Estimated time: 45-60 minutes")
print("=" * 70 + "\n")

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=val_generator,
    validation_steps=val_generator.samples // BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

model.save('final_model.h5')
print("\n✓ Final model saved to 'final_model.h5'")

with open('class_names.pkl', 'wb') as f:
    pickle.dump(class_names, f)
print("✓ Class names saved to 'class_names.pkl'")

# ============================================================================
# PLOT TRAINING HISTORY
# ============================================================================
print("\nGenerating training visualizations...")

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
axes[1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Training history saved to 'training_history.png'")

# ============================================================================
# EVALUATION
# ============================================================================
print("\n" + "=" * 70)
print("EVALUATING MODEL...")
print("=" * 70 + "\n")

val_generator.reset()

print("Generating predictions...")
y_pred = model.predict(val_generator, steps=val_generator.samples // BATCH_SIZE, verbose=1)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = val_generator.classes[:len(y_pred_classes)]

val_loss, val_accuracy = model.evaluate(val_generator, steps=val_generator.samples // BATCH_SIZE, verbose=0)
print(f"\nValidation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")

print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes, target_names=class_names, digits=4))

# ============================================================================
# CONFUSION MATRIX
# ============================================================================
print("\nGenerating confusion matrix...")

cm = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Confusion matrix saved to 'confusion_matrix.png'")

# ============================================================================
# SUMMARY
# ============================================================================
end_time = datetime.now(timeJKT).replace(microsecond=0)
duration = end_time - start_time

print("\n" + "=" * 70)
print("TRAINING COMPLETED!")
print("=" * 70)
print(f"Start Time: {start_time}")
print(f"End Time: {end_time}")
print(f"Total Duration: {duration}")
print(f"\nFinal Results:")
print(f"  - Validation Accuracy: {val_accuracy*100:.2f}%")
print(f"  - Validation Loss: {val_loss:.4f}")
print(f"  - Training samples: {train_generator.samples:,}")
print(f"  - Validation samples: {val_generator.samples:,}")
print(f"\nFiles Generated:")
print("  ✓ best_model.h5")
print("  ✓ final_model.h5")
print("  ✓ class_names.pkl")
print("  ✓ training_history.png")
print("  ✓ confusion_matrix.png")
print("=" * 70)

with open('training_summary.txt', 'w') as f:
    f.write("BISINDO TRAINING SUMMARY (MEDIUM DATASET)\n")
    f.write("=" * 70 + "\n")
    f.write(f"Start Time: {start_time}\n")
    f.write(f"End Time: {end_time}\n")
    f.write(f"Duration: {duration}\n")
    f.write(f"\nDataset:\n")
    f.write(f"  - Training: {train_generator.samples:,}\n")
    f.write(f"  - Validation: {val_generator.samples:,}\n")
    f.write(f"  - Classes: {num_classes}\n")
    f.write(f"\nConfiguration:\n")
    f.write(f"  - Images per class: {IMAGES_PER_CLASS:,}\n")
    f.write(f"  - Image size: {IMG_SIZE}x{IMG_SIZE}\n")
    f.write(f"  - Batch size: {BATCH_SIZE}\n")
    f.write(f"  - Epochs trained: {len(history.history['loss'])}\n")
    f.write(f"\nResults:\n")
    f.write(f"  - Validation Accuracy: {val_accuracy*100:.2f}%\n")
    f.write(f"  - Validation Loss: {val_loss:.4f}\n")

print("\n✓ Training summary saved to 'training_summary.txt'")
print("\nAll done! 🎉")
