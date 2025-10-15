"""
Demonstration script for the refactored music genre classification system.

This script demonstrates the key features without requiring interactive input.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from refactored_genre_classifier import MusicGenreClassifier

def demo_batch_processing():
    """Demonstrate batch processing functionality."""
    print("=== Batch Processing Demo ===")
    
    # Initialize classifier
    classifier = MusicGenreClassifier()
    
    # Check if we have sample audio files
    sample_dir = "Data/genres_original"
    if not os.path.exists(sample_dir):
        print(f"Sample directory not found: {sample_dir}")
        print("Creating a small demo with synthetic data...")
        
        # Create a small demo with synthetic audio
        demo_batch_processing_synthetic()
        return
    
    # Process a few sample files
    print(f"Processing sample files from: {sample_dir}")
    
    # Find a few sample files
    sample_files = []
    for genre_dir in os.listdir(sample_dir):
        genre_path = os.path.join(sample_dir, genre_dir)
        if os.path.isdir(genre_path):
            audio_files = [f for f in os.listdir(genre_path) if f.endswith('.wav')]
            if audio_files:
                sample_files.append(os.path.join(genre_path, audio_files[0]))
                if len(sample_files) >= 3:  # Limit to 3 files for demo
                    break
    
    if sample_files:
        print(f"Found {len(sample_files)} sample files")
        
        # Process each file individually
        results = []
        for i, file_path in enumerate(sample_files):
            print(f"\nProcessing file {i+1}/{len(sample_files)}: {os.path.basename(file_path)}")
            
            try:
                # Load audio file
                import librosa
                audio_data, sr = librosa.load(file_path, sr=22050)
                
                # Extract true genre from directory structure
                true_genre = os.path.basename(os.path.dirname(file_path))
                
                # Predict genre
                result = classifier.predict_genre(audio_data, 'svm')
                
                # Store results
                results.append({
                    'file_name': os.path.basename(file_path),
                    'true_genre': true_genre,
                    'predicted_genre': result['genre'],
                    'confidence': result['confidence'],
                    'latency': result['latency']
                })
                
                print(f"  True Genre: {true_genre}")
                print(f"  Predicted: {result['genre']} (confidence: {result['confidence']:.3f})")
                print(f"  Latency: {result['latency']:.3f}s")
                
            except Exception as e:
                print(f"  Error processing {file_path}: {e}")
        
        # Save results
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv("demo_batch_results.csv", index=False)
        print(f"\nBatch processing completed! Results saved to demo_batch_results.csv")
        print(f"Processed {len(results)} files")
        
    else:
        print("No sample audio files found")
        demo_batch_processing_synthetic()

def demo_batch_processing_synthetic():
    """Demonstrate batch processing with synthetic audio."""
    print("Creating synthetic audio for demonstration...")
    
    # Initialize classifier
    classifier = MusicGenreClassifier()
    
    # Create synthetic audio data
    duration = 3.0  # seconds
    sr = 22050
    t = np.linspace(0, duration, int(sr * duration))
    
    # Create different synthetic audio patterns
    synthetic_files = [
        ("synthetic_blues.wav", np.sin(2 * np.pi * 220 * t) + 0.5 * np.sin(2 * np.pi * 440 * t)),
        ("synthetic_classical.wav", np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)),
        ("synthetic_rock.wav", np.sin(2 * np.pi * 330 * t) + 0.7 * np.sin(2 * np.pi * 660 * t))
    ]
    
    results = []
    for i, (filename, audio_data) in enumerate(synthetic_files):
        print(f"\nProcessing synthetic file {i+1}/{len(synthetic_files)}: {filename}")
        
        # Predict genre
        result = classifier.predict_genre(audio_data, 'svm')
        
        # Store results
        results.append({
            'file_name': filename,
            'true_genre': 'synthetic',
            'predicted_genre': result['genre'],
            'confidence': result['confidence'],
            'latency': result['latency']
        })
        
        print(f"  Predicted: {result['genre']} (confidence: {result['confidence']:.3f})")
        print(f"  Latency: {result['latency']:.3f}s")
    
    # Save results
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv("demo_batch_results.csv", index=False)
    print(f"\nSynthetic batch processing completed! Results saved to demo_batch_results.csv")

def demo_streaming_simulation():
    """Demonstrate streaming simulation functionality."""
    print("\n=== Streaming Simulation Demo ===")
    
    # Initialize classifier
    classifier = MusicGenreClassifier()
    
    # Create synthetic audio for streaming simulation
    print("Creating synthetic audio for streaming simulation...")
    
    duration = 10.0  # seconds
    sr = 22050
    t = np.linspace(0, duration, int(sr * duration))
    
    # Create a more complex synthetic audio pattern
    audio_data = (np.sin(2 * np.pi * 220 * t) + 
                  0.5 * np.sin(2 * np.pi * 440 * t) + 
                  0.3 * np.sin(2 * np.pi * 880 * t))
    
    print(f"Audio duration: {duration} seconds")
    print(f"Chunk duration: {classifier.chunk_duration} seconds")
    print(f"Chunk overlap: {classifier.chunk_overlap} seconds")
    
    # Create chunks
    chunks = classifier.create_audio_chunks(audio_data, sr)
    print(f"Created {len(chunks)} chunks")
    
    # Process chunks
    results = []
    for i, (chunk_audio, start_time, end_time) in enumerate(chunks):
        print(f"\nProcessing chunk {i+1}/{len(chunks)}: {start_time:.1f}s - {end_time:.1f}s")
        
        # Predict genre for this chunk
        result = classifier.predict_genre(chunk_audio, 'svm')
        
        # Store results
        results.append({
            'chunk_index': i,
            'chunk_start': start_time,
            'chunk_end': end_time,
            'predicted_genre': result['genre'],
            'confidence': result['confidence'],
            'latency': result['latency']
        })
        
        print(f"  Chunk prediction: {result['genre']} (confidence: {result['confidence']:.3f})")
        print(f"  Latency: {result['latency']:.3f}s")
    
    # Save results
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv("demo_streaming_results.csv", index=False)
    print(f"\nStreaming simulation completed! Results saved to demo_streaming_results.csv")
    print(f"Processed {len(results)} chunks")

def demo_feature_extraction():
    """Demonstrate feature extraction functionality."""
    print("\n=== Feature Extraction Demo ===")
    
    # Initialize classifier
    classifier = MusicGenreClassifier()
    
    # Create synthetic audio
    duration = 3.0  # seconds
    sr = 22050
    t = np.linspace(0, duration, int(sr * duration))
    audio_data = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    
    print(f"Audio duration: {duration} seconds")
    print(f"Sample rate: {sr} Hz")
    print(f"Audio length: {len(audio_data)} samples")
    
    # Extract features
    features = classifier.extract_features(audio_data, sr)
    
    print(f"Extracted {len(features)} features")
    print(f"Feature vector shape: {features.shape}")
    print(f"First 10 features: {features[:10]}")
    
    # Show feature statistics
    print(f"Feature statistics:")
    print(f"  Mean: {np.mean(features):.4f}")
    print(f"  Std: {np.std(features):.4f}")
    print(f"  Min: {np.min(features):.4f}")
    print(f"  Max: {np.max(features):.4f}")

def demo_model_comparison():
    """Demonstrate model comparison functionality."""
    print("\n=== Model Comparison Demo ===")
    
    # Initialize classifier
    classifier = MusicGenreClassifier()
    
    # Create synthetic audio
    duration = 3.0  # seconds
    sr = 22050
    t = np.linspace(0, duration, int(sr * duration))
    audio_data = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    
    print("Comparing predictions from different models...")
    
    # Test each model
    models = ['random_forest', 'svm', 'cnn']
    results = {}
    
    for model_name in models:
        if classifier.models[model_name] is not None:
            print(f"\nTesting {model_name.upper()} model:")
            result = classifier.predict_genre(audio_data, model_name)
            results[model_name] = result
            
            print(f"  Genre: {result['genre']}")
            print(f"  Confidence: {result['confidence']:.3f}")
            print(f"  Latency: {result['latency']:.3f}s")
        else:
            print(f"\n{model_name.upper()} model not available")
    
    # Compare results
    print("\nModel Comparison Summary:")
    for model_name, result in results.items():
        print(f"  {model_name.upper()}: {result['genre']} (confidence: {result['confidence']:.3f}, latency: {result['latency']:.3f}s)")

def main():
    """Run all demonstrations."""
    print("=== Refactored Music Genre Classification System - Demo ===\n")
    
    # Check if models are available
    models_dir = Path("models")
    if not models_dir.exists():
        print("Error: Models directory not found!")
        print("Please run model training first:")
        print("python src/model_training.py")
        return
    
    # Check for required model files
    required_files = ["scaler.pkl", "label_encoder.pkl", "random_forest_model.pkl", "svm_model.pkl"]
    missing_files = [f for f in required_files if not (models_dir / f).exists()]
    
    if missing_files:
        print(f"Error: Missing model files: {missing_files}")
        print("Please run model training first:")
        print("python src/model_training.py")
        return
    
    print("All required model files found!")
    print("Starting demonstrations...\n")
    
    try:
        # Run demonstrations
        demo_feature_extraction()
        demo_model_comparison()
        demo_batch_processing()
        demo_streaming_simulation()
        
        print("\n" + "=" * 60)
        print("All demonstrations completed successfully!")
        print("=" * 60)
        print("\nGenerated files:")
        print("- demo_batch_results.csv: Batch processing results")
        print("- demo_streaming_results.csv: Streaming simulation results")
        print("\nNext steps:")
        print("1. Run: python src/example_usage.py")
        print("2. Or run: python src/refactored_genre_classifier.py")
        print("3. Or launch the dashboard for interactive use")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
