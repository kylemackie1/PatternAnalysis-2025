"""
Dataset loader for ADNI Alzheimer's Disease MRI data (JPEG format)
Handles loading and preprocessing of brain MRI scan images
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
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