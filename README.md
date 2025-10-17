```markdown
# Music Genre Classification System (Refactored)

## 🎵 Overview

This project provides a **robust, modular music genre classification system** using the GTZAN dataset. It supports:

- **Batch processing** of pre-recorded audio files  
- **Near-real-time streaming simulation**  
- **Interactive dashboard** for live visualization  
- **Traditional ML (SVM, Random Forest)** and **deep learning (CNN, experimental)**

> **Note:** Live microphone input has been removed. The system now processes only pre-recorded audio files.

---

## 🎯 Objectives

- Classify music recordings into 10 genres: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock  
- Provide accurate and efficient predictions using Random Forest and SVM  
- Compare model performance and visualize results  
- Simulate real-time streaming of audio  
- Offer a user-friendly dashboard for monitoring predictions  

---

## 📁 Dataset

Uses **GTZAN Dataset**:

```

Data/
├── genres_original/       # Original audio files
│   ├── blues/
│   ├── classical/
│   ├── country/
│   └── ...
└── features_30_sec.csv    # Pre-extracted features (optional)

````

- 1,000 audio recordings (100 per genre) in WAV format  
- Support for multiple audio formats: `.wav`, `.mp3`, `.flac`, `.m4a`  
- Features: 57-dimensional GTZAN feature vector (spectral, MFCC, chroma, rhythm, energy)  

---

## 🛠️ Models Supported

| Model | Accuracy | Notes |
|-------|----------|-------|
| **SVM** | 71.0% | Best performing, high-dimensional data handling |
| Random Forest | 69.5% | Interpretable, robust to overfitting |
| CNN | 10.0% | Experimental, deep learning approach, limited data |

---

## ⚙️ Installation

### 1. Clone repository

```bash
git clone <repo_url>
cd <repo_folder>
````

### 2. Create virtual environment

```bash
python -m venv venv
```

### 3. Activate environment

* **Windows:** `venv\Scripts\activate`
* **macOS/Linux:** `source venv/bin/activate`

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

Required dependencies:

```
numpy, pandas, librosa, scikit-learn, matplotlib, seaborn, tkinter, joblib, tensorflow (optional)
```

---

## 🚀 Quick Start

### 1. Batch Processing

```python
from src.refactored_genre_classifier import MusicGenreClassifier

classifier = MusicGenreClassifier()
results = classifier.batch_process_songs(
    input_dir="Data/genres_original",
    output_file="batch_results.csv",
    model_name="svm"
)
print(f"Processed {len(results)} files")
```

**Output CSV Columns:**

| Column          | Description               |
| --------------- | ------------------------- |
| file_name       | Audio file name           |
| file_path       | Full path                 |
| true_genre      | Ground truth              |
| predicted_genre | Model prediction          |
| confidence      | Probability (0-1)         |
| latency         | Processing time (seconds) |
| model           | Model used                |

---

### 2. Streaming Simulation (Near-Real-Time)

```python
results = classifier.streaming_simulation(
    file_path="Data/genres_original/blues/blues.00000.wav",
    output_file="streaming_results.csv",
    model_name="svm"
)
print(f"Processed {len(results)} chunks")
```

**Output CSV Columns:**

| Column          | Description         |
| --------------- | ------------------- |
| chunk_start     | Start time (s)      |
| chunk_end       | End time (s)        |
| chunk_index     | Sequence number     |
| true_genre      | Ground truth        |
| predicted_genre | Model prediction    |
| confidence      | Probability (0-1)   |
| latency         | Processing time (s) |
| model           | Model used          |

**Chunk Parameters (configurable):**

```python
classifier.chunk_duration = 3.0    # seconds
classifier.chunk_overlap = 0.5     # seconds
```

---

### 3. Interactive Dashboard

```python
dashboard = classifier.create_dashboard()
dashboard.run()  # Opens GUI
```

Features:

* Genre probability bar chart (live updates)
* Time-series plot of streaming results
* File selection and model comparison
* Start/stop streaming controls

---

## 📊 Performance

| Mode      | Model         | Accuracy | Latency         | Throughput          |
| --------- | ------------- | -------- | --------------- | ------------------- |
| Batch     | SVM           | 71.0%    | 0.5-0.6 s/file  | ~87 files/s         |
| Batch     | Random Forest | 69.5%    | 0.5-0.6 s/file  | ~87 files/s         |
| Streaming | SVM           | 71.0%    | 0.5-0.6 s/chunk | ~0.79 predictions/s |
| Streaming | Random Forest | 69.5%    | 0.5-0.6 s/chunk | ~0.79 predictions/s |

> CNN is experimental and underperforms due to limited training data.

---

## 🔧 Troubleshooting

* **Model Loading Error**: Retrain models

  ```bash
  python src/model_training.py
  ```

* **Missing Dependencies**:

  ```bash
  pip install -r requirements.txt
  ```

* **Audio File Format Issues**: Ensure supported formats (.wav, .mp3, .flac, .m4a) and correct file permissions.

* **Dashboard Not Launching**:

  * Ensure `tkinter` is installed
  * Check display settings for headless systems
  * CLI alternatives: batch or streaming simulation

**Performance Optimization:**

* SSD storage for faster file I/O
* Reduce chunk overlap for faster streaming
* Use SVM for best accuracy, Random Forest for speed

---

## 💻 Project Structure

```
src/
├── refactored_genre_classifier.py  # Main system
├── example_usage.py                 # Sample usage
├── test_refactored_system.py        # Test suite
├── model_training.py                # Model training
├── utils.py                         # Utilities
└── evaluation.py                    # Evaluation scripts

models/
├── scaler.pkl
├── label_encoder.pkl
├── random_forest_model.pkl
├── svm_model.pkl
└── cnn_model.keras

Data/
├── genres_original/                 # GTZAN dataset
└── features_30_sec.csv

results/                              # Output CSVs, visualizations
```

---

## 📝 Contributing

* Modular and well-documented system
* Easy to extend with new models
* Batch, streaming, and dashboard features fully tested

---

## 🏆 Acknowledgments

* GTZAN dataset creators
* Scikit-learn, TensorFlow, Librosa teams
* Open source machine learning community

