import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from PIL import Image
import logging
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # This will help handle truncated images

# Set up logging
logging.basicConfig(
    filename='image_loading_errors.log',
    level=logging.ERROR,
    format='%(asctime)s - %(message)s'
)

class SatelliteDataLoader:
    def __init__(self, data_dir, img_size=(224, 224)):
        """
        Initialize the data loader
        Args:
            data_dir (str): Path to the training_testing_data directory
            img_size (tuple): Size to which images will be resized (height, width)
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.classes = ['Normal', 'Deforestation']
        self.corrupted_images = []
        
    def verify_image(self, image_path):
        """
        Verify if an image file is corrupted
        Args:
            image_path (str): Path to the image file
        Returns:
            bool: True if image is valid, False otherwise
        """
        try:
            with Image.open(image_path) as img:
                img.verify()
            return True
        except Exception as e:
            self.corrupted_images.append((image_path, str(e)))
            logging.error(f"Corrupted image found: {image_path} - Error: {str(e)}")
            return False
    
    def load_and_preprocess_image(self, image_path):
        """
        Load and preprocess a single image with error handling
        Args:
            image_path (str): Path to the image file
        Returns:
            numpy array: Preprocessed image array or None if loading fails
        """
        try:
            img = load_img(image_path, target_size=self.img_size)
            img_array = img_to_array(img)
            return img_array / 255.0  # Normalize pixel values
        except Exception as e:
            self.corrupted_images.append((image_path, str(e)))
            logging.error(f"Failed to load image: {image_path} - Error: {str(e)}")
            return None
            
    def _load_images_from_folder(self, folder_path):
        """
        Load images from a specific folder
        Args:
            folder_path (str): Path to the folder containing images
        Returns:
            tuple: (images array, labels array, coordinates list)
        """
        images = []
        labels = []
        coordinates = []
        skipped_images = 0
        
        for class_idx, class_name in enumerate(self.classes):
            class_path = os.path.join(folder_path, class_name)
            if not os.path.exists(class_path):
                print(f"Warning: Directory not found: {class_path}")
                continue
                
            for img_name in os.listdir(class_path):
                if img_name.endswith('.jpg'):
                    img_path = os.path.join(class_path, img_name)
                    
                    # Verify image integrity
                    if not self.verify_image(img_path):
                        skipped_images += 1
                        continue
                    
                    # Get coordinates from filename
                    coords = img_name.split('.')[0]
                    
                    # Load and preprocess image
                    img_array = self.load_and_preprocess_image(img_path)
                    if img_array is not None:
                        images.append(img_array)
                        labels.append(class_idx)
                        coordinates.append(coords)
                    else:
                        skipped_images += 1
        
        if skipped_images > 0:
            print(f"Warning: Skipped {skipped_images} corrupted or unreadable images")
            print("Check 'image_loading_errors.log' for details")
        
        if not images:
            raise ValueError(f"No valid images found in {folder_path}")
            
        return np.array(images), to_categorical(labels), coordinates
    
    def load_train_data(self):
        """Load training data"""
        print("Loading training data...")
        return self._load_images_from_folder(os.path.join(self.data_dir, 'train'))
    
    def load_test_data(self):
        """Load test data"""
        print("Loading test data...")
        return self._load_images_from_folder(os.path.join(self.data_dir, 'test'))
    
    def get_corrupted_images(self):
        """
        Get list of corrupted images found during loading
        Returns:
            list: List of tuples containing (image_path, error_message)
        """
        return self.corrupted_images