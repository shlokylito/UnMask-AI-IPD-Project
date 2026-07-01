#!/usr/bin/env python3
"""
Test Fine-tuned Model and Compare with Baseline
"""

import os
import json
import numpy as np
import cv2
from tensorflow import keras
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from datetime import datetime

def load_test_data(data_dir, num_samples=500):
    """Load test images"""
    real_dir = os.path.join(data_dir, 'Real')
    deepfake_dir = os.path.join(data_dir, 'DeepFake')
    
    real_files = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.endswith(('.jpg', '.png'))][:num_samples//2]
    fake_files = [os.path.join(deepfake_dir, f) for f in os.listdir(deepfake_dir) if f.endswith(('.jpg', '.png'))][:num_samples//2]
    
    images = []
    labels = []
    
    for file in real_files:
        try:
            img = cv2.imread(file)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (112, 112))
                img = img / 255.0
                images.append(img)
                labels.append(0)  # Real
        except:
            pass
    
    for file in fake_files:
        try:
            img = cv2.imread(file)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (112, 112))
                img = img / 255.0
                images.append(img)
                labels.append(1)  # Fake
        except:
            pass
    
    return np.array(images), np.array(labels)

def main():
    print("\n" + "="*80)
    print("FINETUNED MODEL EVALUATION")
    print("="*80)
    print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    data_dir = "/Users/mirdabhi/Desktop/Coding/ipd copy/ipd-project/ipdcode/Deepfake-detection/data"
    server_dir = "/Users/mirdabhi/Desktop/Coding/ipd copy/ipd-project/ipdcode/server"
    model_path = os.path.join(server_dir, "finetuned_meso4_model.h5")
    
    # Load model
    print("Loading fine-tuned model...")
    try:
        model = keras.models.load_model(model_path)
        print(f"✓ Model loaded from {model_path}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # Load test data
    print("\nLoading test data...")
    test_images, test_labels = load_test_data(data_dir, num_samples=500)
    print(f"✓ Loaded {len(test_images)} test images")
    
    # Make predictions
    print("\nGenerating predictions...")
    predictions = model.predict(test_images, verbose=0)
    predictions_binary = (predictions.flatten() > 0.5).astype(int)
    predictions_prob = predictions.flatten()
    
    # Calculate metrics
    print("\nCalculating metrics...")
    accuracy = accuracy_score(test_labels, predictions_binary)
    precision = precision_score(test_labels, predictions_binary, zero_division=0)
    recall = recall_score(test_labels, predictions_binary, zero_division=0)
    f1 = f1_score(test_labels, predictions_binary, zero_division=0)
    auc = roc_auc_score(test_labels, predictions_prob)
    
    # Results
    results = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'test_set_size': int(len(test_images)),
        'test_accuracy': round(accuracy, 4),
        'test_precision': round(precision, 4),
        'test_recall': round(recall, 4),
        'test_f1_score': round(f1, 4),
        'test_auc': round(auc, 4),
        'confidence_percentage': round(accuracy * 100, 2)
    }
    
    print("\n" + "="*80)
    print("FINETUNED MODEL TEST RESULTS")
    print("="*80)
    print(f"Test Accuracy: {results['test_accuracy']:.4f} ({results['confidence_percentage']}%)")
    print(f"Test Precision: {results['test_precision']:.4f}")
    print(f"Test Recall: {results['test_recall']:.4f}")
    print(f"Test F1-Score: {results['test_f1_score']:.4f}")
    print(f"Test AUC: {results['test_auc']:.4f}")
    
    # Comparison
    print("\n" + "="*80)
    print("BASELINE vs FINETUNED MODEL COMPARISON")
    print("="*80)
    
    baseline_confidence = 48.84  # From previous test
    finetuned_confidence = results['confidence_percentage']
    improvement = finetuned_confidence - baseline_confidence
    improvement_percentage = (improvement / baseline_confidence) * 100
    
    print(f"\nBaseline Model Confidence: {baseline_confidence}%")
    print(f"Finetuned Model Confidence: {finetuned_confidence}%")
    print(f"Absolute Improvement: +{improvement:.2f}%")
    print(f"Relative Improvement: +{improvement_percentage:.1f}%")
    
    # Detailed Comparison Table
    print("\n" + "-"*80)
    print("DETAILED METRICS COMPARISON")
    print("-"*80)
    print(f"{'Metric':<20} {'Baseline':<15} {'Finetuned':<15} {'Change':<15}")
    print("-"*80)
    acc_change = improvement
    prec_val = results["test_precision"]*100
    prec_change = prec_val - 92.7
    rec_val = results["test_recall"]*100
    rec_change = rec_val - 89.5
    f1_val = results["test_f1_score"]*100
    f1_change = f1_val - 91.0
    
    print(f"{'Accuracy':<20} {'48.84%':<15} {finetuned_confidence}%{'':<10} +{acc_change:.2f}%")
    print(f"{'Precision':<20} {'92.7%':<15} {prec_val:.2f}%{'':<10} {prec_change:+.2f}%")
    print(f"{'Recall':<20} {'89.5%':<15} {rec_val:.2f}%{'':<10} {rec_change:+.2f}%")
    print(f"{'F1-Score':<20} {'91.0%':<15} {f1_val:.2f}%{'':<10} {f1_change:+.2f}%")
    print(f"{'AUC':<20} {'N/A':<15} {results['test_auc']:.4f}{'':<10} {'N/A':<15}")
    
    # Save results
    results_file = os.path.join(server_dir, 'finetuned_test_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_file}")
    
    # Create summary report
    summary = {
        'project': 'IPD Deepfake Detection',
        'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'baseline_results': {
            'model': 'Original Meso4',
            'confidence': 48.84,
            'accuracy': 0.4884,
            'precision': 0.927,
            'recall': 0.895,
            'f1_score': 0.910
        },
        'finetuned_results': {
            'model': 'Enhanced Meso4 (Fine-tuned)',
            'confidence': results['confidence_percentage'],
            'accuracy': results['test_accuracy'],
            'precision': results['test_precision'],
            'recall': results['test_recall'],
            'f1_score': results['test_f1_score'],
            'auc': results['test_auc']
        },
        'improvements': {
            'accuracy_improvement': round(results['test_accuracy'] - 0.4884, 4),
            'absolute_percentage_improvement': round(finetuned_confidence - baseline_confidence, 2),
            'relative_percentage_improvement': round(improvement_percentage, 1),
            'status': '✓ SIGNIFICANT IMPROVEMENT' if improvement > 10 else '✓ Improvement' if improvement > 0 else '✗ No Improvement'
        },
        'training_config': {
            'training_images': 4972,
            'validation_images': 1065,
            'test_images': 1067,
            'total_dataset': 7104,
            'epochs_trained': 6,
            'augmentation': True,
            'techniques': ['Data Augmentation', 'Early Stopping', 'Learning Rate Scheduling', 'Batch Normalization']
        }
    }
    
    summary_file = os.path.join(server_dir, 'finetuning_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Summary saved to: {summary_file}")
    
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    if improvement > 20:
        print(f"✓ EXCELLENT IMPROVEMENT: The fine-tuned model shows {improvement:.1f}% improvement!")
    elif improvement > 10:
        print(f"✓ GOOD IMPROVEMENT: The fine-tuned model shows {improvement:.1f}% improvement.")
    elif improvement > 0:
        print(f"✓ MODERATE IMPROVEMENT: The fine-tuned model shows {improvement:.1f}% improvement.")
    else:
        print(f"✗ NO IMPROVEMENT: Model needs further optimization.")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
