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
    

def prepare_data_from_directory(data_dir, val_size=0.15, random_state=42):
    """
    Prepare train/val/test splits from ADNI directory structure
    
    Expected directory structure:
        data_dir/
            ├── train/
            │   ├── AD/
            │   │   ├── scan1.jpg
            │   │   └── ...
            │   └── NC/
            │       ├── scan1.jpg
            │       └── ...
            └── test/
                ├── AD/
                │   ├── scan1.jpg
                │   └── ...
                └── NC/
                    ├── scan1.jpg
                    └── ...
    
    Args:
        data_dir: Path to root data directory (should contain train/ and test/)
        val_size: Fraction of training data to use for validation
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_paths, train_labels, val_paths, val_labels, test_paths, test_labels)
    """
    data_dir = Path(data_dir)
    
    # Check if directory structure is correct
    train_dir = data_dir / 'train'
    test_dir = data_dir / 'test'
    
    if not train_dir.exists() or not test_dir.exists():
        raise ValueError(f"Expected 'train' and 'test' subdirectories in {data_dir}")
    
    # Collect training files (support jpg, jpeg, png)
    train_nc_dir = train_dir / 'NC'
    train_ad_dir = train_dir / 'AD'
    
    train_nc_files = []
    train_ad_files = []
    
    if train_nc_dir.exists():
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            train_nc_files.extend(list(train_nc_dir.glob(ext)))
    
    if train_ad_dir.exists():
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            train_ad_files.extend(list(train_ad_dir.glob(ext)))
    
    # Collect test files
    test_nc_dir = test_dir / 'NC'
    test_ad_dir = test_dir / 'AD'
    
    test_nc_files = []
    test_ad_files = []
    
    if test_nc_dir.exists():
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            test_nc_files.extend(list(test_nc_dir.glob(ext)))
    
    if test_ad_dir.exists():
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            test_ad_files.extend(list(test_ad_dir.glob(ext)))
    
    print(f"\nData Summary:")
    print(f"  Training set:")
    print(f"    NC (Normal) cases: {len(train_nc_files)}")
    print(f"    AD cases: {len(train_ad_files)}")
    print(f"    Total: {len(train_nc_files) + len(train_ad_files)}")
    print(f"  Test set:")
    print(f"    NC (Normal) cases: {len(test_nc_files)}")
    print(f"    AD cases: {len(test_ad_files)}")
    print(f"    Total: {len(test_nc_files) + len(test_ad_files)}")
    
    # Check if we have data
    if len(train_nc_files) + len(train_ad_files) == 0:
        raise ValueError(f"No training data found in {train_dir}")
    if len(test_nc_files) + len(test_ad_files) == 0:
        raise ValueError(f"No test data found in {test_dir}")
    
    # Prepare training data (will split into train/val)
    train_all_files = train_nc_files + train_ad_files
    train_all_labels = [0] * len(train_nc_files) + [1] * len(train_ad_files)  # 0=NC, 1=AD
    
    # Convert to strings
    train_all_files = [str(f) for f in train_all_files]
    
    # Split training data into train and validation
    train_files, val_files, train_labels, val_labels = train_test_split(
        train_all_files, train_all_labels,
        test_size=val_size,
        random_state=random_state,
        stratify=train_all_labels
    )
    
    # Prepare test data
    test_files = test_nc_files + test_ad_files
    test_labels = [0] * len(test_nc_files) + [1] * len(test_ad_files)  # 0=NC, 1=AD
    test_files = [str(f) for f in test_files]
    
    # Calculate class balance
    total_samples = len(train_files) + len(val_files) + len(test_files)
    total_ad = sum(train_labels) + sum(val_labels) + sum(test_labels)
    
    print(f"\nClass balance: {total_ad/total_samples:.2%} AD, {1-total_ad/total_samples:.2%} NC")
    
    print(f"\nSplit Summary:")
    print(f"  Train: {len(train_files)} ({sum(train_labels)} AD, {len(train_labels)-sum(train_labels)} NC)")
    print(f"  Val:   {len(val_files)} ({sum(val_labels)} AD, {len(val_labels)-sum(val_labels)} NC)")
    print(f"  Test:  {len(test_files)} ({sum(test_labels)} AD, {len(test_labels)-sum(test_labels)} NC)")
    
    return train_files, train_labels, val_files, val_labels, test_files, test_labels