import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

DATASET_DIR = r"C:\tomato_doctor\dataset-tomatoes"
MODEL_PATH = "model/tomato_model.h5"

if not os.path.exists(MODEL_PATH):
    print("[ERROR] Model file not found! Train the model first.")
else:
    model = tf.keras.models.load_model(MODEL_PATH)
    val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    val_generator = val_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation',
        seed=42,
        shuffle=False
    )

    loss, accuracy = model.evaluate(val_generator)
    print("\n" + "="*40)
    print(f"📊 Validation Loss:     {loss:.4f}")
    print(f"🎯 Validation Accuracy: {accuracy * 100:.2f}%")
    print("="*40)