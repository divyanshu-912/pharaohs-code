import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import numpy as np
from backend.models.captcha_model import CaptchaModel
from backend.utils.data_generator import CaptchaDataGenerator
from backend.config import Config

def verify_dataset():
    train_dir = "data/captcha_dataset/train"
    val_dir = "data/captcha_dataset/validation"
    test_dir = "data/captcha_dataset/test"
    
    print("Dataset Statistics:")
    print(f"Training images: {len(os.listdir(train_dir))}")
    print(f"Validation images: {len(os.listdir(val_dir))}")
    print(f"Test images: {len(os.listdir(test_dir))}")


def train():
    # Verify dataset
    verify_dataset()
    
    try:
        # Create data generators
        train_generator = CaptchaDataGenerator(
            Config.TRAIN_DATA_DIR,
            batch_size=Config.BATCH_SIZE,
            augment=True,
            shuffle=True
        )
        
        val_generator = CaptchaDataGenerator(
            Config.VAL_DATA_DIR,
            batch_size=Config.BATCH_SIZE,
            augment=False,
            shuffle=False
        )
        
        # Create directories for checkpoints
        os.makedirs('checkpoints', exist_ok=True)
        
        # Initialize callbacks
        callbacks = [
            # Model checkpoint
            tf.keras.callbacks.ModelCheckpoint(
                'checkpoints/model_{epoch:02d}.keras',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
           
            # Reduce learning rate on plateau
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1
            )
        ]
        

        
        # Initialize and train model
        print("Initializing model...")
        model = CaptchaModel()
        
        print("Starting training...")
        history = model.train(
            train_generator,
            val_generator,
            epochs=Config.EPOCHS,
            callbacks=callbacks
        )
        
        # Save the final model
        model.save_model()
        print("Training completed. Model saved.")
        return history
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    # Enable memory growth for GPU if available
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            print("GPU memory growth enabled")
        except RuntimeError as e:
            print(f"Error enabling GPU memory growth: {str(e)}")
    
    # Start training
    history = train()
