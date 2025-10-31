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