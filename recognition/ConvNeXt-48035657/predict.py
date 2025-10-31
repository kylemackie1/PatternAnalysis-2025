"""
Prediction script for Alzheimer's Disease classification
Loads a trained model and evaluates on test set
"""

import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, classification_report
)

from modules import create_model
from dataset import create_dataloaders


def plot_confusion_matrix(cm, save_path, class_names=['Normal', 'AD']):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def plot_roc_curve(labels, probs, save_path):
    """Plot and save ROC curve"""
    fpr, tpr, thresholds = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"ROC curve saved to {save_path}")
    plt.close()


def plot_attention_analysis(attentions, labels, predictions, save_path):
    """Plot attention weight analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Separate by class
    nc_attentions = np.array([att for att, label in zip(attentions, labels) if label == 0])
    ad_attentions = np.array([att for att, label in zip(attentions, labels) if label == 1])

    # Average attention per slice for each class
    if len(nc_attentions) > 0:
        nc_avg = nc_attentions.mean(axis=0)
        axes[0, 0].bar(range(len(nc_avg)), nc_avg, color='steelblue', alpha=0.7)
        axes[0, 0].set_title('Average Attention - Normal Control', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Slice Index')
        axes[0, 0].set_ylabel('Average Attention Weight')
        axes[0, 0].grid(True, alpha=0.3)

    if len(ad_attentions) > 0:
        ad_avg = ad_attentions.mean(axis=0)
        axes[0, 1].bar(range(len(ad_avg)), ad_avg, color='coral', alpha=0.7)
        axes[0, 1].set_title('Average Attention - Alzheimer\'s Disease', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Slice Index')
        axes[0, 1].set_ylabel('Average Attention Weight')
        axes[0, 1].grid(True, alpha=0.3)

    # Compare NC vs AD attention patterns
    if len(nc_attentions) > 0 and len(ad_attentions) > 0:
        axes[1, 0].plot(nc_avg, label='Normal', linewidth=2, marker='o')
        axes[1, 0].plot(ad_avg, label='AD', linewidth=2, marker='s')
        axes[1, 0].set_title('Attention Pattern Comparison', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Slice Index')
        axes[1, 0].set_ylabel('Average Attention Weight')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    # Distribution of max attention weights
    max_attentions = [att.max() for att in attentions]
    axes[1, 1].hist(max_attentions, bins=30, color='purple', alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Distribution of Max Attention Weights', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Max Attention Weight')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Attention analysis saved to {save_path}")
    plt.close()


def evaluate_model(model, test_loader, device):
    """
    Evaluate model on test set

    Returns:
        Dictionary with all metrics and predictions
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    all_attentions = []

    print("\nRunning predictions on test set...")
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output, attention = model(data)
            probs = torch.softmax(output, dim=1)
            preds = output.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(target.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_attentions.extend(attention.cpu().numpy())

            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{len(test_loader)} batches")

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


def print_results(metrics):
    """Print detailed test results"""
    print("\n" + "=" * 70)
    print("TEST SET RESULTS")
    print("=" * 70)
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
    print("=" * 70)

    # Check if target accuracy reached
    if metrics['accuracy'] >= 0.8:
        print(f"\n✓ SUCCESS! Target accuracy of 0.8 achieved: {metrics['accuracy']:.4f}")
    else:
        print(f"\n✗ Target accuracy not reached. Got {metrics['accuracy']:.4f}, need ≥ 0.8")


def save_results(metrics, output_dir):
    """Save metrics and plots"""
    os.makedirs(output_dir, exist_ok=True)

    # Save metrics to JSON
    metrics_to_save = {
        'accuracy': float(metrics['accuracy']),
        'precision': float(metrics['precision']),
        'recall': float(metrics['recall']),
        'f1_score': float(metrics['f1_score']),
        'auc': float(metrics['auc']),
        'confusion_matrix': metrics['confusion_matrix'].tolist()
    }

    metrics_path = os.path.join(output_dir, 'test_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_to_save, f, indent=4)
    print(f"\nMetrics saved to {metrics_path}")

    # Plot confusion matrix
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(metrics['confusion_matrix'], cm_path)

    # Plot ROC curve
    roc_path = os.path.join(output_dir, 'roc_curve.png')
    plot_roc_curve(metrics['labels'], metrics['probabilities'], roc_path)

    # Plot attention analysis
    attention_path = os.path.join(output_dir, 'attention_analysis.png')
    plot_attention_analysis(
        metrics['attentions'],
        metrics['labels'],
        metrics['predictions'],
        attention_path
    )

    # Save predictions to CSV
    predictions_path = os.path.join(output_dir, 'predictions.csv')
    with open(predictions_path, 'w') as f:
        f.write("patient_idx,true_label,predicted_label,probability_ad,correct\n")
        for i, (label, pred, prob) in enumerate(zip(
            metrics['labels'],
            metrics['predictions'],
            metrics['probabilities']
        )):
            correct = "Yes" if label == pred else "No"
            f.write(f"{i},{label},{pred},{prob:.4f},{correct}\n")
    print(f"Predictions saved to {predictions_path}")


def main(args):
    """Main prediction function"""
    print("=" * 70)
    print("ALZHEIMER'S DISEASE CLASSIFICATION - PREDICTION")
    print("=" * 70)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"\nUsing device: {device}")

    # Load model checkpoint
    print(f"\nLoading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)

    # Create model
    model = create_model(
        num_classes=2,
        dropout=args.dropout,
        num_slices=args.num_slices
    )

    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded successfully (epoch {checkpoint.get('epoch', 'unknown')})")
        print(f"  Best validation accuracy: {checkpoint.get('val_acc', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
        print("Model loaded successfully")

    model = model.to(device)
    model.eval()

    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load test data
    print("\nLoading test data...")
    _, _, test_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=(args.img_size, args.img_size),
        augment=False,
        use_validation=False,  # We only need test set
        num_slices=args.num_slices
    )

    print(f"Test set: {len(test_loader.dataset)} patients")

    # Evaluate
    metrics = evaluate_model(model, test_loader, device)

    # Print results
    print_results(metrics)

    # Save results
    output_dir = args.output_dir or os.path.join(
        os.path.dirname(args.model_path),
        'predictions'
    )
    save_results(metrics, output_dir)

    print(f"\n✓ All results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Predict Alzheimer\'s Disease using trained model'
    )

    # Required arguments
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to saved model checkpoint (.pth file)')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to data directory')

    # Model parameters
    parser.add_argument('--num_slices', type=int, default=15,
                       help='Number of slices per patient')
    parser.add_argument('--dropout', type=float, default=0.4,
                       help='Dropout rate (should match training)')

    # Data parameters
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for prediction')
    parser.add_argument('--img_size', type=int, default=224,
                       help='Image size')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')

    # Output parameters
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save results (default: model_dir/predictions)')
    parser.add_argument('--cpu', action='store_true',
                       help='Use CPU even if GPU is available')

    args = parser.parse_args()

    # Validate model path
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")

    # Validate data directory
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")

    # Run prediction
    main(args)