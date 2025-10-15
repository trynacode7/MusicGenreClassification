"""
Example usage of the refactored music genre classification system.

This script demonstrates how to use the MusicGenreClassifier for:
1. Batch processing of entire songs
2. Streaming simulation with chunked processing
3. Real-time dashboard visualization
"""

import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from refactored_genre_classifier import MusicGenreClassifier

def example_batch_processing():
    """Example of batch processing entire songs."""
    print("=== Batch Processing Example ===")
    
    # Initialize classifier
    classifier = MusicGenreClassifier()
    
    # Process songs from a directory
    input_dir = "Data/genres_original"  # GTZAN dataset directory
    output_file = "batch_results.csv"
    model_name = "svm"  # Use SVM model (best performing)
    
    if os.path.exists(input_dir):
        print(f"Processing songs from: {input_dir}")
        results = classifier.batch_process_songs(input_dir, output_file, model_name)
        
        print(f"\nBatch processing completed!")
        print(f"Results saved to: {output_file}")
        print(f"Processed {len(results)} files")
        
        # Show sample results
        print("\nSample results:")
        print(results[['file_name', 'true_genre', 'predicted_genre', 'confidence']].head())
    else:
        print(f"Directory not found: {input_dir}")
        print("Please ensure the GTZAN dataset is available in the Data/genres_original directory")

def example_streaming_simulation():
    """Example of streaming simulation with chunked processing."""
    print("\n=== Streaming Simulation Example ===")
    
    # Initialize classifier
    classifier = MusicGenreClassifier()
    
    # Find a sample audio file
    sample_files = [
        "Data/genres_original/blues/blues.00000.wav",
        "Data/genres_original/classical/classical.00000.wav",
        "Data/genres_original/rock/rock.00000.wav"
    ]
    
    file_path = None
    for sample_file in sample_files:
        if os.path.exists(sample_file):
            file_path = sample_file
            break
    
    if file_path:
        print(f"Running streaming simulation on: {file_path}")
        
        # Run streaming simulation
        results = classifier.streaming_simulation(
            file_path=file_path,
            output_file="streaming_results.csv",
            model_name="svm",
            chunk_delay=0.1  # 100ms delay between chunks
        )
        
        print(f"\nStreaming simulation completed!")
        print(f"Results saved to: streaming_results.csv")
        print(f"Processed {len(results)} chunks")
        
        # Show sample results
        print("\nSample chunk results:")
        print(results[['chunk_start', 'chunk_end', 'predicted_genre', 'confidence']].head())
    else:
        print("No sample audio files found")
        print("Please ensure the GTZAN dataset is available")

def example_dashboard():
    """Example of launching the real-time dashboard."""
    print("\n=== Real-time Dashboard Example ===")
    
    # Initialize classifier
    classifier = MusicGenreClassifier()
    
    # Find a sample audio file
    sample_files = [
        "Data/genres_original/blues/blues.00000.wav",
        "Data/genres_original/classical/classical.00000.wav",
        "Data/genres_original/rock/rock.00000.wav"
    ]
    
    file_path = None
    for sample_file in sample_files:
        if os.path.exists(sample_file):
            file_path = sample_file
            break
    
    if file_path:
        print(f"Launching dashboard with: {file_path}")
        print("The dashboard will show:")
        print("- Genre probability bar chart")
        print("- Streaming results over time")
        print("- Real-time updates as chunks are processed")
        
        # Launch dashboard
        dashboard = classifier.create_dashboard()
        dashboard.run()
    else:
        print("No sample audio files found")
        print("Please ensure the GTZAN dataset is available")

def main():
    """Main function demonstrating all features."""
    print("Music Genre Classification - Example Usage")
    print("=" * 50)
    
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
    print("\nChoose an example to run:")
    print("1. Batch processing example")
    print("2. Streaming simulation example")
    print("3. Real-time dashboard example")
    print("4. Run all examples")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == '1':
        example_batch_processing()
    elif choice == '2':
        example_streaming_simulation()
    elif choice == '3':
        example_dashboard()
    elif choice == '4':
        example_batch_processing()
        example_streaming_simulation()
        example_dashboard()
    else:
        print("Invalid choice. Please run the script again.")

if __name__ == "__main__":
    main()
