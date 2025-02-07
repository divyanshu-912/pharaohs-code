class Config:
    # Image parameters
    IMAGE_HEIGHT = 50
    IMAGE_WIDTH = 200
    CHANNELS = 1
    
    # Training parameters
    BATCH_SIZE = 32
    EPOCHS = 15
    LEARNING_RATE = 0.001
    
    # CAPTCHA parameters
    CAPTCHA_LENGTH = 5
    # Updated charset to include both cases
    CHARSET = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    CHAR_COUNT = len(CHARSET)
    
    # Directories
    TRAIN_DATA_DIR = 'data/captcha_dataset/train'
    VAL_DATA_DIR = 'data/captcha_dataset/validation'
    TEST_DATA_DIR = 'data/captcha_dataset/test'
    MODEL_SAVE_PATH = 'backend/models/captcha_model.keras'
