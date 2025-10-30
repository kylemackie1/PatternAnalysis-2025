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
    def __init__(self, file_paths, labels, transform=None, img_size=(224, 224),
                 num_slices=20):
        """
        Args:
            file_paths: List of paths to JPEG files
            labels: List of labels (0 for NC/Normal, 1 for AD)
            transform: Data augmentation transforms
            img_size: Target image size (height, width)
            num_slices: Expected number of slices per patient
        """
        self.transform = transform
        self.img_size = img_size
        self.num_slices = num_slices

        # Group files by patient ID
        self.patients = {}
        for path, label in zip(file_paths, labels):
            # Extract patient ID (everything before first underscore)
            patient_id = Path(path).stem.split('_')[0]

            if patient_id not in self.patients:
                self.patients[patient_id] = {'files': [], 'label': label}
            self.patients[patient_id]['files'].append(path)

        # Convert to list for indexing
        self.patient_ids = list(self.patients.keys())

        print(f"Dataset initialized with {len(self.patient_ids)} patients")
        print(f"  - Total images: {len(file_paths)}")
        print(f"  - Avg slices per patient: {len(file_paths)/len(self.patient_ids):.1f}")
        print(f"  - Image size: {img_size}")

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        """Load and preprocess all slices for a single patient"""
        patient_id = self.patient_ids[idx]
        patient_data = self.patients[patient_id]
        file_paths = sorted(patient_data['files'])  # Sort to ensure consistent order
        label = patient_data['label']

        slices = []
        for img_path in file_paths:
            try:
                # Load image
                image = Image.open(img_path)

                # Convert to grayscale if needed
                if image.mode != 'L':
                    image = image.convert('L')

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
                image = np.zeros(self.img_size, dtype=np.float32)

            slices.append(image)

        # Stack slices into array
        slices_array = np.stack(slices, axis=0)  # (num_slices, height, width)

        # Apply augmentation to each slice
        if self.transform:
            augmented_slices = []
            for slice_2d in slices_array:
                aug_slice = self.transform(slice_2d)
                augmented_slices.append(aug_slice.squeeze(0))  # Remove channel dim
            slices_array = torch.stack(augmented_slices, dim=0)  # (num_slices, height, width)
        else:
            # Convert to tensor
            slices_array = torch.from_numpy(slices_array).float()

        label = torch.tensor(label, dtype=torch.long)

        return slices_array, label


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

    # Check directory structure
    train_dir = data_dir / 'train'
    test_dir = data_dir / 'test'
    if not train_dir.exists() or not test_dir.exists():
        raise ValueError(f"Expected 'train' and 'test' subdirectories in {data_dir}")

    def collect_images(label_dir):
        """Collect image paths and subject IDs."""
        files, subjects = [], []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            for f in label_dir.glob(ext):
                files.append(f)
                subject_id = f.stem.split('_')[0]  # everything before the underscore
                subjects.append(subject_id)
        return files, subjects

    # ---- Collect training images ----
    train_nc_files, train_nc_subjects = collect_images(train_dir / 'NC')
    train_ad_files, train_ad_subjects = collect_images(train_dir / 'AD')

    # ---- Collect test images ----
    test_nc_files, test_nc_subjects = collect_images(test_dir / 'NC')
    test_ad_files, test_ad_subjects = collect_images(test_dir / 'AD')

    print(f"\nData Summary:")
    print(f"  Training set: NC={len(train_nc_files)}, AD={len(train_ad_files)}, Total={len(train_nc_files)+len(train_ad_files)}")
    print(f"  Test set: NC={len(test_nc_files)}, AD={len(test_ad_files)}, Total={len(test_nc_files)+len(test_ad_files)}")

    # --- Combine and prepare for splitting ---
    train_all_files = train_nc_files + train_ad_files
    train_all_labels = [0] * len(train_nc_files) + [1] * len(train_ad_files)
    train_all_subjects = train_nc_subjects + train_ad_subjects

    # Get unique subjects and map labels (AD=1, NC=0)
    subject_to_label = {}
    for subj, label in zip(train_all_subjects, train_all_labels):
        # If multiple scans exist for same subject, ensure consistent label
        if subj not in subject_to_label:
            subject_to_label[subj] = label

    unique_subjects = list(subject_to_label.keys())
    unique_labels = [subject_to_label[s] for s in unique_subjects]

    # --- Split by subject ---
    train_subs, val_subs = train_test_split(
        unique_subjects,
        test_size=val_size,
        random_state=random_state,
        stratify=unique_labels
    )

    # --- Assign images based on subject split ---
    train_files, val_files, train_labels, val_labels = [], [], [], []
    for f, subj, label in zip(train_all_files, train_all_subjects, train_all_labels):
        if subj in train_subs:
            train_files.append(str(f))
            train_labels.append(label)
        elif subj in val_subs:
            val_files.append(str(f))
            val_labels.append(label)

    # --- Prepare test set ---
    test_files = [str(f) for f in test_nc_files + test_ad_files]
    test_labels = [0] * len(test_nc_files) + [1] * len(test_ad_files)

    # --- Print summary ---
    total_samples = len(train_files) + len(val_files) + len(test_files)
    total_ad = sum(train_labels) + sum(val_labels) + sum(test_labels)

    print(f"\nClass balance: {total_ad/total_samples:.2%} AD, {1 - total_ad/total_samples:.2%} NC")
    print(f"\nSplit Summary:")
    print(f"  Train: {len(train_files)} ({sum(train_labels)} AD, {len(train_labels)-sum(train_labels)} NC)")
    print(f"  Val:   {len(val_files)} ({sum(val_labels)} AD, {len(val_labels)-sum(val_labels)} NC)")
    print(f"  Test:  {len(test_files)} ({sum(test_labels)} AD, {len(test_labels)-sum(test_labels)} NC)")

    print(f"\nUnique subjects -> Train: {len(set(train_subs))}, Val: {len(set(val_subs))}")
    return train_files, train_labels, val_files, val_labels, test_files, test_labels


def create_dataloaders(data_dir, batch_size=8, num_workers=4, 
                       img_size=(224, 224), augment=True):
    """
    Create train, validation, and test dataloaders
    
    Args:
        data_dir: Path to data directory with train/ and test/ subdirectories
        batch_size: Batch size for training
        num_workers: Number of parallel data loading workers
        img_size: Target image size (height, width)
        augment: Whether to apply data augmentation to training set
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    print("\n" + "=" * 70)
    print("PREPARING DATALOADERS")
    print("=" * 70)
    
    # Prepare data splits
    train_files, train_labels, val_files, val_labels, test_files, test_labels = \
        prepare_data_from_directory(data_dir)
    
    # Create augmentation for training
    augmentation = DataAugmentation(
        rotation_range=0,
        horizontal_flip=False,
        vertical_flip=False,
        brightness_range=0.25,
        contrast_range=0.25,
        noise_std=0.03
    ) if augment else None
    
    # Create datasets
    train_dataset = ADNIDataset(
        train_files, train_labels,
        transform=augmentation,
        img_size=img_size
    )
    
    val_dataset = ADNIDataset(
        val_files, val_labels,
        transform=None,  # No augmentation for validation
        img_size=img_size
    )
    
    test_dataset = ADNIDataset(
        test_files, test_labels,
        transform=None,  # No augmentation for testing
        img_size=img_size
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # Drop last incomplete batch
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\nDataloaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    print("=" * 70 + "\n")
    
    return train_loader, val_loader, test_loader
