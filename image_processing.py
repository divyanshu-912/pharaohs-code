import cv2
import numpy as np
from backend.config import Config

def preprocess_image(image_file):
    try:
        if isinstance(image_file, str):
            img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        else:
            img_array = np.frombuffer(image_file.read(), np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            raise ValueError("Failed to load image")
        
        # Resize
        img = cv2.resize(img, (Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT))
        
        # Apply adaptive thresholding
        img = cv2.adaptiveThreshold(
            img, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        
        # Denoise
        img = cv2.fastNlMeansDenoising(img)
        
        # Normalize
        img = img.astype('float32') / 255.0
        
        # Add channel dimension
        img = np.expand_dims(img, axis=-1)
        
        # Add batch dimension
        return np.expand_dims(img, axis=0)
    
    except Exception as e:
        print(f"Error in image preprocessing: {e}")
        raise
