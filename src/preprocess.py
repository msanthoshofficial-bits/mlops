import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def get_train_val_generators(base_dir, img_size=(224, 224), batch_size=32):
    preprocessing_function = tf.keras.applications.mobilenet_v2.preprocess_input

    # Using validation_split since data is not pre-split
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocessing_function,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2 
    )

    # For validation, we don't want augmentation, but we need preprocessing
    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocessing_function,
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        base_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training'
    )

    validation_generator = test_datagen.flow_from_directory(
        base_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation'
    )

    return train_generator, validation_generator

if __name__ == "__main__":
    base_dir = r"C:\dev\mlops\Dataset\PetImages"
    print(f"Using local dataset at: {base_dir}")
