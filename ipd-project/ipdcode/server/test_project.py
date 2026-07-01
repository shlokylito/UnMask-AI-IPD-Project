#!/usr/bin/env python3
"""
Comprehensive test script for IPD Deepfake Detection Project
Tests video, audio, and frame analysis using MTCNN, MediaPipe, MobileNetV2, and Wav2Vec2
"""

import os
import json
import cv2
import numpy as np
import time
from pathlib import Path
from datetime import datetime

# Import analysis modules
from face import detect_face_distortion
from frame import detect_frame_anomalies
from audio import analyze_video
from senti import analyze_video_sentiment

def get_video_duration(video_path):
    """Get video duration in seconds"""
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        return frame_count / fps if fps > 0 else 0
    except:
        return 0

def get_video_info(video_path):
    """Extract basic video information"""
    try:
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return {
            'width': width,
            'height': height,
            'fps': fps,
            'frames': frame_count,
            'duration': frame_count / fps if fps > 0 else 0
        }
    except Exception as e:
        print(f"Error reading video {video_path}: {e}")
        return None

def process_test_video(video_path):
    """Process a single video and return analysis results"""
    print(f"\n{'='*80}")
    print(f"Processing: {os.path.basename(video_path)}")
    print(f"{'='*80}")
    
    video_info = get_video_info(video_path)
    if not video_info:
        return None
    
    print(f"Video Info: {video_info['width']}x{video_info['height']}@{video_info['fps']}fps, Duration: {video_info['duration']:.2f}s")
    
    results = {
        'filename': os.path.basename(video_path),
        'duration': video_info['duration'],
        'resolution': f"{video_info['width']}x{video_info['height']}",
        'fps': video_info['fps'],
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    try:
        # Face Distortion Detection (MTCNN + MediaPipe)
        print("\n[1/3] Analyzing face distortion using MTCNN & MediaPipe...")
        start = time.time()
        try:
            total_frames, distorted_faces = detect_face_distortion(video_path, skip_frames=5)
            face_time = time.time() - start
            distortion_ratio = (distorted_faces / max(total_frames, 1)) * 100
            results['total_frames'] = total_frames
            results['distorted_faces'] = distorted_faces
            results['distortion_ratio'] = round(distortion_ratio, 2)
            results['face_analysis_time'] = round(face_time, 2)
            print(f"   ✓ Processed {total_frames} frames, {distorted_faces} distorted ({distortion_ratio:.1f}%)")
            print(f"   ✓ Time: {face_time:.2f}s")
        except Exception as e:
            print(f"   ✗ Face analysis failed: {e}")
            results['face_analysis_error'] = str(e)
        
        # Frame Anomaly Detection
        print("\n[2/3] Analyzing frame anomalies...")
        start = time.time()
        try:
            total_frames_processed, abnormal_frames = detect_frame_anomalies(video_path, skip_frames=5)
            frame_time = time.time() - start
            anomaly_ratio = (abnormal_frames / max(total_frames_processed, 1)) * 100
            results['frames_processed'] = total_frames_processed
            results['abnormal_frames'] = abnormal_frames
            results['anomaly_ratio'] = round(anomaly_ratio, 2)
            results['frame_analysis_time'] = round(frame_time, 2)
            print(f"   ✓ Detected {abnormal_frames} anomalies in {total_frames_processed} frames ({anomaly_ratio:.1f}%)")
            print(f"   ✓ Time: {frame_time:.2f}s")
        except Exception as e:
            print(f"   ✗ Frame analysis failed: {e}")
            results['frame_analysis_error'] = str(e)
        
        # Audio-Visual Analysis (Wav2Vec2 + Audio Analysis)
        print("\n[3/3] Analyzing audio-visual synchronization using Wav2Vec2...")
        start = time.time()
        try:
            metrics, face_detection_rate = analyze_video(video_path)
            audio_time = time.time() - start
            results['cosine_similarity'] = round(metrics['cosine_similarity'], 4)
            results['mismatch_score'] = round(metrics['mismatch_score'], 4)
            results['euclidean_distance'] = round(metrics['euclidean_distance'], 4)
            results['face_detection_rate'] = round(face_detection_rate, 4)
            results['audio_analysis_time'] = round(audio_time, 2)
            print(f"   ✓ Cosine Similarity: {results['cosine_similarity']}")
            print(f"   ✓ Mismatch Score: {results['mismatch_score']}")
            print(f"   ✓ Face Detection Rate: {results['face_detection_rate']}")
            print(f"   ✓ Time: {audio_time:.2f}s")
        except Exception as e:
            print(f"   ✗ Audio analysis failed: {e}")
            results['audio_analysis_error'] = str(e)
        
        # Calculate composite scores
        try:
            distortion_ratio = results.get('distortion_ratio', 50) / 100
            anomaly_ratio = results.get('anomaly_ratio', 50) / 100
            mismatch_score = results.get('mismatch_score', 0.5)
            
            # Face authenticity score (0-100)
            face_score = max(0, 100 * (1 - distortion_ratio) * results.get('face_detection_rate', 0))
            
            # Frame consistency score (0-100)
            frame_score = max(0, 100 * (1 - anomaly_ratio))
            
            # Audio-visual sync score (0-100)
            av_sync_score = max(0, 100 * (1 - mismatch_score))
            
            # Overall confidence (average of all scores)
            overall_confidence = round((face_score + frame_score + av_sync_score) / 3, 2)
            
            # Prediction
            if overall_confidence > 70:
                prediction = "REAL"
            elif overall_confidence > 40:
                prediction = "UNCERTAIN"
            else:
                prediction = "DEEPFAKE"
            
            results['face_authenticity_score'] = round(face_score, 2)
            results['frame_consistency_score'] = round(frame_score, 2)
            results['av_sync_score'] = round(av_sync_score, 2)
            results['overall_confidence'] = overall_confidence
            results['prediction'] = prediction
            
            print(f"\n[RESULTS]")
            print(f"  Face Authenticity Score: {face_score:.2f}%")
            print(f"  Frame Consistency Score: {frame_score:.2f}%")
            print(f"  AV Sync Score: {av_sync_score:.2f}%")
            print(f"  Overall Confidence: {overall_confidence}%")
            print(f"  Prediction: {prediction}")
            
        except Exception as e:
            print(f"Error calculating scores: {e}")
            results['scoring_error'] = str(e)
        
        return results
        
    except Exception as e:
        print(f"Fatal error processing {video_path}: {e}")
        results['error'] = str(e)
        return results

def main():
    """Main test function"""
    server_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Find test videos
    test_videos = [
        os.path.join(server_dir, "barackobama.mp4"),
        os.path.join(server_dir, "000471.mp4"),
        os.path.join(server_dir, "uploads/happy.mp4"),
    ]
    
    # Filter to existing files only
    existing_videos = [v for v in test_videos if os.path.exists(v)]
    
    if not existing_videos:
        print("No test videos found!")
        return
    
    print("\n" + "="*80)
    print("IPD DEEPFAKE DETECTION PROJECT - COMPREHENSIVE TEST RESULTS")
    print("="*80)
    print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Videos to Test: {len(existing_videos)}")
    print(f"Technologies: MTCNN, MediaPipe, MobileNetV2, Wav2Vec2")
    
    all_results = []
    
    for video_path in existing_videos:
        result = process_test_video(video_path)
        if result:
            all_results.append(result)
    
    # Generate summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    
    if all_results:
        # Print header
        print(f"\n{'File':<25} {'Duration':<10} {'Prediction':<12} {'Confidence':<12} {'Face Score':<12} {'Frame Score':<12} {'AV Sync':<10}")
        print("-" * 105)
        
        for result in all_results:
            filename = result.get('filename', '')[:24]
            duration = f"{result.get('duration', 0):.1f}s"
            prediction = result.get('prediction', 'ERROR')
            confidence = f"{result.get('overall_confidence', 0)}%"
            face_score = f"{result.get('face_authenticity_score', 0):.1f}%"
            frame_score = f"{result.get('frame_consistency_score', 0):.1f}%"
            av_sync = f"{result.get('av_sync_score', 0):.1f}%"
            
            print(f"{filename:<25} {duration:<10} {prediction:<12} {confidence:<12} {face_score:<12} {frame_score:<12} {av_sync:<10}")
    
    # Save detailed results to JSON
    output_file = os.path.join(server_dir, "test_results.json")
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✓ Detailed results saved to: {output_file}")
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
