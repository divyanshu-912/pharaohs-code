import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from backend.config import Config

class CaptchaModel:
    def __init__(self):
        self.model = None
        self.model = self._build_model()
    
    def _build_model(self):
        inputs = layers.Input(shape=(Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH, Config.CHANNELS))
        
        # First Convolutional Block
        x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        # Second Convolutional Block
        x = layers.Conv2D(64, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        # Third Convolutional Block
        x = layers.Conv2D(128, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        # Dense Layers
        x = layers.Flatten()(x)
        x = layers.Dense(512)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.5)(x)
        
        # Output Layer
        x = layers.Dense(Config.CAPTCHA_LENGTH * Config.CHAR_COUNT)(x)
        x = layers.Reshape((Config.CAPTCHA_LENGTH, Config.CHAR_COUNT))(x)
        outputs = layers.Activation('softmax')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def train(self, train_generator, val_generator, epochs=Config.EPOCHS, callbacks=None):
        if callbacks is None:
            callbacks = []
            
        return self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
    
    def predict(self, image):
        return self.model.predict(image, verbose=0)
    
    def decode_prediction(self, prediction):
        result = ''
        try:
            for i in range(Config.CAPTCHA_LENGTH):
                char_idx = np.argmax(prediction[0, i])
                if 0 <= char_idx < len(Config.CHARSET):
                    result += Config.CHARSET[char_idx]
        except Exception as e:
            print(f"Error in decoding: {e}")
        return result
    
    def save_model(self, path=None):
        if path is None:
            path = Config.MODEL_SAVE_PATH
        self.model.save(path)
    
    def load_model(self, path):
        self.model = tf.keras.models.load_model(path)
