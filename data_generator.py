import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from backend.config import Config
from backend.utils.image_processing import preprocess_image

class CaptchaDataGenerator(Sequence):
    def __init__(self, data_dir, batch_size=32, augment=False, shuffle=True):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith('.png')]
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.floor(len(self.image_files) / self.batch_size))
    
    @property
    def num_batches(self):
        return self.__len__()
    
    def __getitem__(self, idx):
        batch_files = self.image_files[idx * self.batch_size:(idx + 1) * self.batch_size]
        X = np.zeros((self.batch_size, Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH, Config.CHANNELS))
        y = np.zeros((self.batch_size, Config.CAPTCHA_LENGTH, Config.CHAR_COUNT))
        
        for i, filename in enumerate(batch_files):
            img_path = os.path.join(self.data_dir, filename)
            with open(img_path, 'rb') as f:
                img = preprocess_image(f)
            X[i] = img[0]
            
            label = filename.split('.')[0]
            for j, char in enumerate(label):
                char_idx = Config.CHARSET.index(char)
                y[i, j, char_idx] = 1
        
        return tf.convert_to_tensor(X), tf.convert_to_tensor(y)
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.image_files)
