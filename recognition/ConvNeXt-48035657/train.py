"""
Training script for Alzheimer's Disease classification
Handles training, validation, testing, and model checkpointing
"""

import os
import json
import time
import argparse
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report
)
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

from modules import create_model
from dataset import create_dataloaders


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    def __init__(self, patience=10, min_delta=0.0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self.mode == 'min' and score > self.best_score - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        elif self.mode == 'max' and score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


class Trainer:
    """Trainer class for model training and evaluation"""
    def __init__(self, model, train_loader, val_loader, test_loader, 
                 criterion, optimizer, scheduler, device, save_dir='checkpoints'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = save_dir
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Tracking metrics
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.learning_rates = []
        
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            output, _ = self.model(data)  # Ignore attention weights during training
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Track metrics
            running_loss += loss.item()
            preds = output.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(target.cpu().numpy())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx + 1}/{len(self.train_loader)}, "
                      f"Loss: {loss.item():.4f}")
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)

                output, _ = self.model(data)  # Ignore attention weights
                loss = self.criterion(output, target)

                running_loss += loss.item()
                probs = torch.softmax(output, dim=1)
                preds = output.argmax(dim=1).cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(target.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)

        return epoch_loss, epoch_acc
    
    def test(self):
        """Test the model and return detailed metrics"""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        all_attentions = []
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)

                output, attention = self.model(data)
                probs = torch.softmax(output, dim=1)
                preds = output.argmax(dim=1).cpu().numpy()

                all_preds.extend(preds)
                all_labels.extend(target.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
                all_attentions.extend(attention.cpu().numpy())

        # Calculate metrics
        acc = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='binary')
        recall = recall_score(all_labels, all_preds, average='binary')
        f1 = f1_score(all_labels, all_preds, average='binary')
        auc = roc_auc_score(all_labels, all_probs)
        cm = confusion_matrix(all_labels, all_preds)

        metrics = {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'confusion_matrix': cm,
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs,
            'attentions': all_attentions
        }

        return metrics

    def train(self, num_epochs, early_stopping_patience=15):
        """Train the model for multiple epochs"""
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        early_stopping = EarlyStopping(patience=early_stopping_patience, mode='max')

        start_time = time.time()

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)

            # Train
            train_loss, train_acc = self.train_epoch()

            # Validate
            val_loss, val_acc = self.validate()

            # Update learning rate
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']

            # Track metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            self.learning_rates.append(current_lr)

            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"Learning Rate: {current_lr:.6f}")

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.save_checkpoint('best_model.pth', epoch, val_acc, val_loss)
                print(f"✓ New best model saved! Val Acc: {val_acc:.4f}")

            # Early stopping check
            early_stopping(val_acc)
            if early_stopping.early_stop:
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                break

        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time / 60:.2f} minutes")
        print(f"Best validation accuracy: {self.best_val_acc:.4f}")

        # Save training history
        self.save_training_history()

        # Plot training curves
        self.plot_training_curves()

        return self.best_val_acc

    def save_checkpoint(self, filename, epoch, val_acc, val_loss):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs
        }
        path = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, path)

    def load_checkpoint(self, filename):
        """Load model checkpoint"""
        path = os.path.join(self.save_dir, filename)
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint

    def save_training_history(self):
        """Save training history to JSON"""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'learning_rates': self.learning_rates,
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss
        }
        path = os.path.join(self.save_dir, 'training_history.json')
        with open(path, 'w') as f:
            json.dump(history, f, indent=4)

    def plot_training_curves(self):
        """Plot and save training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Loss curves
        axes[0, 0].plot(self.train_losses, label='Train Loss', linewidth=2)
        axes[0, 0].plot(self.val_losses, label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Loss', fontsize=12)
        axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0, 0].legend(fontsize=11)
        axes[0, 0].grid(True, alpha=0.3)

        # Accuracy curves
        axes[0, 1].plot(self.train_accs, label='Train Acc', linewidth=2)
        axes[0, 1].plot(self.val_accs, label='Val Acc', linewidth=2)
        axes[0, 1].axhline(y=0.8, color='r', linestyle='--', label='Target (0.8)', linewidth=2)
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('Accuracy', fontsize=12)
        axes[0, 1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        axes[0, 1].legend(fontsize=11)
        axes[0, 1].grid(True, alpha=0.3)

        # Learning rate
        axes[1, 0].plot(self.learning_rates, linewidth=2, color='green')
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('Learning Rate', fontsize=12)
        axes[1, 0].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)

        # Overfitting check
        axes[1, 1].plot(np.array(self.train_accs) - np.array(self.val_accs),
                       linewidth=2, color='purple')
        axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=1)
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('Train Acc - Val Acc', fontsize=12)
        axes[1, 1].set_title('Overfitting Monitor', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, 'training_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nTraining curves saved to {save_path}")
        plt.close()


def plot_confusion_matrix(cm, save_path):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'AD'],
                yticklabels=['Normal', 'AD'],
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def print_test_results(metrics):
    """Print detailed test results"""
    print("\n" + "=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print(f"AUC-ROC:   {metrics['auc']:.4f}")
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    print("\nClassification Report:")
    print(classification_report(
        metrics['labels'],
        metrics['predictions'],
        target_names=['Normal', 'AD'],
        digits=4
    ))
    print("=" * 60)


def main(args):
    """Main training function"""
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Create dataloaders
    print("\nLoading data...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=(args.img_size, args.img_size),
        augment=True
    )
    
    # Create model
    print("\nCreating model...")
    model = create_model(
        num_classes=2,
        dropout=args.dropout
    )
    model = model.to(device)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    if args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
    else:
        scheduler = CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6
        )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=args.save_dir
    )
    
    # Train model
    best_val_acc = trainer.train(
        num_epochs=args.epochs,
        early_stopping_patience=args.patience
    )
    
    # Load best model and test
    print("\nLoading best model for testing...")
    trainer.load_checkpoint('best_model.pth')
    
    print("\nEvaluating on test set...")
    test_metrics = trainer.test()
    
    # Print and save results
    print_test_results(test_metrics)
    
    # Plot confusion matrix
    cm_path = os.path.join(args.save_dir, 'confusion_matrix.png')
    plot_confusion_matrix(test_metrics['confusion_matrix'], cm_path)
    
    # Save test metrics
    metrics_to_save = {
        'accuracy': float(test_metrics['accuracy']),
        'precision': float(test_metrics['precision']),
        'recall': float(test_metrics['recall']),
        'f1_score': float(test_metrics['f1_score']),
        'auc': float(test_metrics['auc']),
        'confusion_matrix': test_metrics['confusion_matrix'].tolist()
    }
    
    metrics_path = os.path.join(args.save_dir, 'test_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_to_save, f, indent=4)
    
    print(f"\nTest metrics saved to {metrics_path}")
    
    # Check if target accuracy reached
    if test_metrics['accuracy'] >= 0.8:
        print(f"\n✓ SUCCESS! Target accuracy of 0.8 achieved: {test_metrics['accuracy']:.4f}")
    else:
        print(f"\n✗ Target accuracy not reached. Got {test_metrics['accuracy']:.4f}, need ≥ 0.8")
        print("Consider: training longer, adjusting hyperparameters, or using more data")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Alzheimer\'s Disease Classifier')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to data directory')
    parser.add_argument('--num_slices', type=int, default=16,
                       help='Number of slices per volume')
    parser.add_argument('--img_size', type=int, default=224,
                       help='Image size (height and width)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience')
    
    # Model parameters
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout rate')
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['plateau', 'cosine'],
                       help='Learning rate scheduler')
    
    # Other parameters
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    
    args = parser.parse_args()
    
    # Create save directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    args.save_dir = os.path.join(args.save_dir, f'run_{timestamp}')
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(args.save_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    print("Configuration:")
    print(json.dumps(vars(args), indent=2))
    
    # Run training
    main(args)