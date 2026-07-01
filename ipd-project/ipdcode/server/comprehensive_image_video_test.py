#!/usr/bin/env python3
"""
Comprehensive Image and Video Testing with Fine-tuned Model
"""

import os
import cv2
import numpy as np
from tensorflow import keras
import json
from datetime import datetime

def load_finetuned_model(model_path):
    """Load the fine-tuned model"""
    try:
        model = keras.models.load_model(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def predict_image(image_path, model):
    """Predict single image"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None, "Error: Could not read image"
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (112, 112))
        img_normalized = img / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        prediction = model.predict(img_batch, verbose=0)[0][0]
        confidence = prediction * 100
        
        if prediction > 0.5:
            label = "DEEPFAKE"
        else:
            label = "REAL"
        
        return confidence, label
    except Exception as e:
        return None, f"Error: {str(e)}"

def predict_video_frames(video_path, model, sample_frames=10):
    """Predict on video frames"""
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            return None, "Error: Could not read video"
        
        frame_indices = np.linspace(0, total_frames - 1, sample_frames, dtype=int)
        predictions = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (112, 112))
            frame_normalized = frame / 255.0
            frame_batch = np.expand_dims(frame_normalized, axis=0)
            
            pred = model.predict(frame_batch, verbose=0)[0][0]
            predictions.append(pred)
        
        cap.release()
        
        if not predictions:
            return None, "Error: No frames processed"
        
        avg_confidence = np.mean(predictions) * 100
        
        if avg_confidence > 50:
            label = "DEEPFAKE"
        else:
            label = "REAL"
        
        return avg_confidence, label
    except Exception as e:
        return None, f"Error: {str(e)}"

def main():
    print("\n" + "="*100)
    print("IPD DEEPFAKE DETECTION - IMAGE & VIDEO TESTING WITH FINE-TUNED MODEL")
    print("="*100)
    print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    model_path = "/Users/mirdabhi/Desktop/Coding/ipd copy/ipd-project/ipdcode/server/finetuned_meso4_model.h5"
    data_dir = "/Users/mirdabhi/Desktop/Coding/ipd copy/ipd-project/ipdcode/Deepfake-detection/data"
    video_dir = "/Users/mirdabhi/Desktop/Coding/ipd copy/ipd-project/ipdcode/server"
    
    # Load model
    print("Loading fine-tuned model...")
    model = load_finetuned_model(model_path)
    if model is None:
        print("Failed to load model")
        return
    print("✓ Model loaded successfully\n")
    
    # Test Images
    print("="*100)
    print("IMAGE TESTING")
    print("="*100)
    
    real_dir = os.path.join(data_dir, 'Real')
    fake_dir = os.path.join(data_dir, 'DeepFake')
    
    # Get sample images
    real_images = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.endswith(('.jpg', '.png'))][:5]
    fake_images = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.endswith(('.jpg', '.png'))][:5]
    
    image_results = []
    
    print(f"\n{'Image Type':<15} {'Filename':<30} {'Confidence':<15} {'Prediction':<15} {'Status':<10}")
    print("-" * 100)
    
    for img_path in real_images:
        conf, pred = predict_image(img_path, model)
        filename = os.path.basename(img_path)
        
        if conf is not None:
            status = "✓" if pred == "REAL" else "✗"
            print(f"{'REAL':<15} {filename:<30} {conf:.2f}%{'':<9} {pred:<15} {status:<10}")
            image_results.append({
                'type': 'REAL',
                'filename': filename,
                'confidence': round(conf, 2),
                'prediction': pred,
                'correct': pred == 'REAL'
            })
        else:
            print(f"{'REAL':<15} {filename:<30} {'ERROR':<15} {pred:<15}")
    
    for img_path in fake_images:
        conf, pred = predict_image(img_path, model)
        filename = os.path.basename(img_path)
        
        if conf is not None:
            status = "✓" if pred == "DEEPFAKE" else "✗"
            print(f"{'DEEPFAKE':<15} {filename:<30} {conf:.2f}%{'':<9} {pred:<15} {status:<10}")
            image_results.append({
                'type': 'DEEPFAKE',
                'filename': filename,
                'confidence': round(conf, 2),
                'prediction': pred,
                'correct': pred == 'DEEPFAKE'
            })
        else:
            print(f"{'DEEPFAKE':<15} {filename:<30} {'ERROR':<15} {pred:<15}")
    
    # Calculate image accuracy
    correct_images = sum(1 for r in image_results if r['correct'])
    image_accuracy = (correct_images / len(image_results)) * 100 if image_results else 0
    print(f"\nImage Test Accuracy: {correct_images}/{len(image_results)} ({image_accuracy:.1f}%)")
    
    # Test Videos
    print("\n" + "="*100)
    print("VIDEO TESTING")
    print("="*100)
    
    video_files = [
        os.path.join(video_dir, "barackobama.mp4"),
        os.path.join(video_dir, "000471.mp4"),
        os.path.join(video_dir, "uploads/happy.mp4")
    ]
    
    video_files = [v for v in video_files if os.path.exists(v)]
    
    video_results = []
    
    print(f"\n{'Video File':<30} {'Duration':<12} {'Frames Sampled':<15} {'Confidence':<15} {'Prediction':<15}")
    print("-" * 100)
    
    for video_path in video_files:
        filename = os.path.basename(video_path)
        
        # Get video info
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        cap.release()
        
        # Predict
        conf, pred = predict_video_frames(video_path, model, sample_frames=10)
        
        if conf is not None:
            print(f"{filename:<30} {duration:.1f}s{'':<6} {'10':<15} {conf:.2f}%{'':<9} {pred:<15}")
            video_results.append({
                'filename': filename,
                'duration': round(duration, 1),
                'frames_sampled': 10,
                'confidence': round(conf, 2),
                'prediction': pred
            })
        else:
            print(f"{filename:<30} {duration:.1f}s{'':<6} {'10':<15} {'ERROR':<15} {pred:<15}")
    
    # Summary Statistics
    print("\n" + "="*100)
    print("SUMMARY STATISTICS")
    print("="*100)
    
    summary = {
        'test_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'image_tests': {
            'total_images': len(image_results),
            'correct_predictions': correct_images,
            'accuracy_percentage': round(image_accuracy, 2)
        },
        'video_tests': {
            'total_videos': len(video_results),
            'videos_tested': video_results
        },
        'model_info': {
            'model_type': 'Enhanced Meso4 (Fine-tuned)',
            'model_file': 'finetuned_meso4_model.h5'
        }
    }
    
    print(f"\nImage Testing:")
    print(f"  - Total Images Tested: {len(image_results)}")
    print(f"  - Correct Predictions: {correct_images}")
    print(f"  - Accuracy: {image_accuracy:.1f}%")
    
    print(f"\nVideo Testing:")
    print(f"  - Total Videos Tested: {len(video_results)}")
    for video in video_results:
        print(f"    • {video['filename']}: {video['confidence']:.1f}% → {video['prediction']}")
    
    # Save results
    results_file = "/Users/mirdabhi/Desktop/Coding/ipd copy/ipd-project/ipdcode/server/comprehensive_test_results.json"
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Results saved to: comprehensive_test_results.json")
    print("="*100 + "\n")

if __name__ == "__main__":
    main()
