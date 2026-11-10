import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

# Configuration
DATASET_DIR = r"C:\tomato_doctor\dataset-tomatoes"
MODEL_SAVE_PATH = "model/tomato_model.h5"
LABELS_SAVE_PATH = "model/class_indices.json"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_STAGE_1 = 8
EPOCHS_STAGE_2 = 8

os.makedirs("model", exist_ok=True)

# Data Augmentation Generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.2
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

print("[INFO] Loading dataset from folder...")
train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    seed=42,
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    seed=42,
    shuffle=False
)

# Automated Class Weight Calculation for Imbalanced Classes
total_samples = len(train_generator.classes)
num_classes = train_generator.num_classes
class_counts = np.bincount(train_generator.classes)
class_weights = {i: total_samples / (num_classes * count) for i, count in enumerate(class_counts)}
print(f"[INFO] Computed balanced class weights: {class_weights}")

# Build Architecture
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
outputs = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=outputs)

# Stage 1: Feature Extraction
print("\n[INFO] Stage 1: Training classification head...")
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_stage1 = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True, verbose=1)
]

model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_STAGE_1,
    class_weight=class_weights,
    callbacks=callbacks_stage1
)

# Stage 2: Fine-Tuning Top Layers
print("\n[INFO] Stage 2: Fine-tuning MobileNetV2 layers for high precision...")
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_stage2 = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, verbose=1),
    ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True, verbose=1)
]

model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_STAGE_2,
    class_weight=class_weights,
    callbacks=callbacks_stage2
)

# Save Class Indices Mapping
class_labels = {v: k for k, v in train_generator.class_indices.items()}
with open(LABELS_SAVE_PATH, "w") as f:
    json.dump(class_labels, f)

print(f"\n[SUCCESS] Model successfully saved to {MODEL_SAVE_PATH}")
print(f"[SUCCESS] Class indices saved to {LABELS_SAVE_PATH}")