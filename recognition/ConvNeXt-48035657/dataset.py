"""
Dataset loader for ADNI Alzheimer's Disease MRI data (JPEG format)
Handles loading and preprocessing of brain MRI scan images
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
import random
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import warnings
warnings.filterwarnings('ignore')


class ADNIDataset(Dataset):
    """
    Dataset class for ADNI brain MRI scans (JPEG images)
    Each sample is a 2D brain scan image
    """
    def __init__(self, file_paths, labels, transform=None, img_size=(224, 224)):
        """
        Args:
            file_paths: List of paths to JPEG files
            labels: List of labels (0 for NC/Normal, 1 for AD)
            transform: Data augmentation transforms
            img_size: Target image size (height, width)
        """
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        self.img_size = img_size
        
        print(f"Dataset initialized with {len(file_paths)} images")
        print(f"  - Image size: {img_size}")
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        """Load and preprocess a single MRI image"""
        img_path = self.file_paths[idx]
        label = self.labels[idx]
        
        try:
            # Load image
            image = Image.open(img_path)
            
            # Convert to numpy array
            image = np.array(image, dtype=np.float32)
            
            # Normalize to [0, 1]
            image = image / 255.0
            
            # Resize if needed
            if image.shape != self.img_size:
                image = Image.fromarray((image * 255).astype(np.uint8))
                image = image.resize(self.img_size, Image.BILINEAR)
                image = np.array(image, dtype=np.float32) / 255.0
            
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return zeros if loading fails
            image = np.zeros(self.img_size, dtype=np.float32)
        
        # Apply augmentation
        if self.transform:
            image = self.transform(image)
        else:
            # Convert to tensor (add channel dimension)
            image = torch.from_numpy(image).unsqueeze(0).float()
        
        label = torch.tensor(label, dtype=torch.long)
        
        return image, label


class DataAugmentation:
    """
    Data augmentation transforms for MRI images
    Applied during training to increase data diversity
    """
    def __init__(self, rotation_range=15, horizontal_flip=True, 
                 vertical_flip=False, brightness_range=0.2, 
                 contrast_range=0.2, noise_std=0.02):
        """
        Args:
            rotation_range: Max rotation angle in degrees
            horizontal_flip: Enable random horizontal flips
            vertical_flip: Enable random vertical flips
            brightness_range: Random brightness adjustment range
            contrast_range: Random contrast adjustment range
            noise_std: Standard deviation of Gaussian noise
        """
        self.rotation_range = rotation_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.noise_std = noise_std
    
    def __call__(self, image):
        """
        Apply random augmentations to image
        
        Args:
            image: 2D numpy array (height, width) in range [0, 1]
            
        Returns:
            Augmented image as tensor (1, height, width)
        """
        # Convert to PIL for transformations
        image_pil = Image.fromarray((image * 255).astype(np.uint8), mode='L')
        
        # Random rotation
        if self.rotation_range > 0:
            angle = random.uniform(-self.rotation_range, self.rotation_range)
            image_pil = image_pil.rotate(angle, resample=Image.BILINEAR, fillcolor=0)
        
        # Random horizontal flip
        if self.horizontal_flip and random.random() > 0.5:
            image_pil = image_pil.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Random vertical flip
        if self.vertical_flip and random.random() > 0.5:
            image_pil = image_pil.transpose(Image.FLIP_TOP_BOTTOM)
        
        # Convert back to numpy
        image = np.array(image_pil, dtype=np.float32) / 255.0
        
        # Brightness adjustment
        if self.brightness_range > 0:
            factor = random.uniform(1 - self.brightness_range, 1 + self.brightness_range)
            image = image * factor
            image = np.clip(image, 0, 1)
        
        # Contrast adjustment
        if self.contrast_range > 0:
            mean = image.mean()
            factor = random.uniform(1 - self.contrast_range, 1 + self.contrast_range)
            image = (image - mean) * factor + mean
            image = np.clip(image, 0, 1)
        
        # Gaussian noise
        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, image.shape)
            image = image + noise
            image = np.clip(image, 0, 1)
        
        # Convert to tensor and add channel dimension
        image = torch.from_numpy(image).unsqueeze(0).float()
        
        return image