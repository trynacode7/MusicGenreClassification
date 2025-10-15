# Refactored Music Genre Classification System

## Overview

This refactored system provides a comprehensive music genre classification solution with **batch processing**, **streaming simulation**, and **real-time dashboard visualization**. The system works exclusively with pre-recorded audio files (no live microphone input) and supports both traditional machine learning models and deep learning approaches.

## Key Features

### 1. **Batch Processing**
- Process entire songs at once
- Support for multiple audio formats (.wav, .mp3, .flac, .m4a)
- CSV output with detailed results
- Automatic genre extraction from directory structure

### 2. **Near-Real-Time Streaming Simulation**
- Split audio files into overlapping chunks (3-second chunks with 0.5s overlap)
- Sequential processing to simulate real-time behavior
- Configurable chunk duration and overlap
- CSV output with chunk-level results

### 3. **Real-Time Dashboard**
- Interactive GUI with genre probability visualization
- Live updates during streaming simulation
- Bar charts showing genre probabilities
- Time-series plots of streaming results
- File selection and model comparison

### 4. **Modular Architecture**
- Separate functions for feature extraction, prediction, and UI
- Clean separation of concerns
- Easy to extend and modify
- Pre-trained model support

## Models Supported

- **Random Forest**: 69.5% accuracy, robust and interpretable
- **SVM**: 71.0% accuracy (best performing), effective for high-dimensional data
- **CNN**: 10.0% accuracy (experimental), deep learning approach

## Installation

### Prerequisites

```bash
pip install -r requirements.txt
```

### Required Dependencies

```
numpy
pandas
librosa
scikit-learn
matplotlib
seaborn
tkinter
joblib
tensorflow (optional, for CNN)
```

## Usage

### 1. **Command Line Interface**

```bash
python src/refactored_genre_classifier.py
```

This launches an interactive menu with options for:
- Batch processing
- Streaming simulation
- Real-time dashboard
- File selection and model comparison

### 2. **Example Usage**

```bash
python src/example_usage.py
```

This demonstrates all features with sample data.

### 3. **Programmatic Usage**

```python
from src.refactored_genre_classifier import MusicGenreClassifier

# Initialize classifier
classifier = MusicGenreClassifier()

# Batch processing
results = classifier.batch_process_songs(
    input_dir="path/to/audio/files",
    output_file="batch_results.csv",
    model_name="svm"
)

# Streaming simulation
results = classifier.streaming_simulation(
    file_path="path/to/audio/file.wav",
    output_file="streaming_results.csv",
    model_name="svm"
)

# Launch dashboard
dashboard = classifier.create_dashboard()
dashboard.run()
```

## File Structure

```
src/
├── refactored_genre_classifier.py  # Main classifier system
├── example_usage.py                 # Example usage script
├── model_training.py               # Model training (existing)
├── utils.py                        # Utility functions (existing)
└── evaluation.py                  # Model evaluation (existing)

models/
├── scaler.pkl                      # Feature scaler
├── label_encoder.pkl              # Label encoder
├── random_forest_model.pkl         # Random Forest model
├── svm_model.pkl                   # SVM model
└── cnn_model.keras                 # CNN model (optional)

Data/
├── genres_original/                # GTZAN dataset
│   ├── blues/
│   ├── classical/
│   ├── country/
│   └── ...
└── features_30_sec.csv           # Pre-extracted features
```

## Output Files

### Batch Processing Output (`batch_results.csv`)

| Column | Description |
|--------|-------------|
| file_name | Name of the audio file |
| file_path | Full path to the audio file |
| true_genre | True genre (from directory structure) |
| predicted_genre | Predicted genre |
| confidence | Prediction confidence (0-1) |
| latency | Processing time in seconds |
| model | Model used for prediction |

### Streaming Simulation Output (`streaming_results.csv`)

| Column | Description |
|--------|-------------|
| file_name | Name of the audio file |
| chunk_start | Start time of chunk (seconds) |
| chunk_end | End time of chunk (seconds) |
| chunk_index | Index of chunk in sequence |
| true_genre | True genre (from directory structure) |
| predicted_genre | Predicted genre for this chunk |
| confidence | Prediction confidence (0-1) |
| latency | Processing time in seconds |
| model | Model used for prediction |

## Configuration

### Chunk Parameters

```python
# Modify these parameters in the classifier
classifier.chunk_duration = 3.0    # Chunk duration in seconds
classifier.chunk_overlap = 0.5     # Overlap between chunks in seconds
```

### Model Selection

```python
# Available models
models = ["random_forest", "svm", "cnn"]

# SVM is recommended for best accuracy (71.0%)
# Random Forest for interpretability (69.5%)
# CNN for experimental deep learning (10.0%)
```

## Dashboard Features

### Real-Time Visualization

1. **Genre Probability Bar Chart**
   - Shows probability distribution across all genres
   - Updates in real-time as chunks are processed
   - Color-coded for easy interpretation

2. **Streaming Results Plot**
   - Time-series plot of confidence scores
   - Shows genre predictions over time
   - Annotated with genre labels

3. **Interactive Controls**
   - File selection with browse dialog
   - Model selection dropdown
   - Start/stop streaming controls
   - Batch processing button

## Performance Metrics

### Batch Processing
- **Throughput**: ~87 files/second (optimal batch size)
- **Accuracy**: 71.0% (SVM), 69.5% (Random Forest)
- **Latency**: 0.5-0.6 seconds per file

### Streaming Simulation
- **Chunk Processing**: 3-second chunks with 0.5s overlap
- **Real-time Factor**: 0.1s delay between chunks
- **Latency**: 0.5-0.6 seconds per chunk
- **Throughput**: ~0.79 predictions/second

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   ```bash
   # Retrain models if needed
   python src/model_training.py
   ```

2. **Missing Dependencies**
   ```bash
   # Install required packages
   pip install -r requirements.txt
   ```

3. **Audio File Format Issues**
   - Supported formats: .wav, .mp3, .flac, .m4a
   - Ensure files are not corrupted
   - Check file permissions

4. **Dashboard Not Launching**
   - Ensure tkinter is installed
   - Check display settings (for headless systems)
   - Try running without GUI: use command-line interface

### Performance Optimization

1. **For Batch Processing**
   - Use SSD storage for faster file I/O
   - Increase batch size for better throughput
   - Use SVM model for best accuracy

2. **For Streaming Simulation**
   - Reduce chunk overlap for faster processing
   - Use Random Forest for faster inference
   - Adjust chunk delay for realistic simulation

## Examples

### Example 1: Batch Process GTZAN Dataset

```python
classifier = MusicGenreClassifier()
results = classifier.batch_process_songs(
    input_dir="Data/genres_original",
    output_file="gtzan_results.csv",
    model_name="svm"
)
print(f"Processed {len(results)} files")
```

### Example 2: Streaming Simulation

```python
classifier = MusicGenreClassifier()
results = classifier.streaming_simulation(
    file_path="Data/genres_original/blues/blues.00000.wav",
    output_file="blues_streaming.csv",
    model_name="svm"
)
print(f"Processed {len(results)} chunks")
```

### Example 3: Launch Dashboard

```python
classifier = MusicGenreClassifier()
dashboard = classifier.create_dashboard()
dashboard.run()  # Opens interactive GUI
```

## Contributing

This refactored system is designed to be:
- **Modular**: Easy to extend with new models
- **Configurable**: Adjustable parameters for different use cases
- **User-friendly**: Both CLI and GUI interfaces
- **Well-documented**: Clear code structure and comments

## License

This project uses the GTZAN dataset. Please refer to the dataset's original license terms.

## Acknowledgments

- GTZAN dataset creators
- Scikit-learn, TensorFlow, and Librosa development teams
- Open source machine learning community
