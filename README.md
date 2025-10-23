Perfect — here’s your **clean, final, submission-ready `README.md`** after removing **Batch Processing** and **Streaming Simulation**, keeping only the **Streamlit dashboard**, and adding **`features_3_sec.csv`** to the dataset section.

Everything is polished for clarity, structure, and professionalism 👇

---

```markdown
# 🎵 Music Genre Classification from Audio Signals

**Project ID:** 10  
**Team:**  
- Diya Prakash — SRN: PES2UG23CS184  
- Erin Joseph — SRN: PES2UG23CS186  

---

## 📘 Overview

This project implements a **robust, modular music genre classification system** using the **GTZAN dataset**.  
It supports:

- 📊 **Interactive Streamlit dashboard** for real-time visualization  
- ⏱️ **Near-real-time streaming simulation** via the dashboard  
- 🧠 **Traditional ML (SVM, Random Forest)** and **experimental CNN (deep learning)**  

> **Note:** The system processes only **pre-recorded audio files**. Live microphone input has been removed.

---

## 🎯 Objectives

- Classify music into **10 genres:** blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock  
- Provide accurate and efficient predictions using **Random Forest** and **SVM**  
- Enable near-real-time simulation and live visualization through a dashboard  
- Offer an intuitive, interactive interface for audio-based genre classification  

---

## 📁 Dataset — GTZAN Music Genre Dataset

**Structure:**

```

Data/
├── genres_original/       # Original 30-sec audio files
│   ├── blues/
│   ├── classical/
│   ├── country/
│   ├── disco/
│   ├── hiphop/
│   ├── jazz/
│   ├── metal/
│   ├── pop/
│   ├── reggae/
│   └── rock/
├── features_30_sec.csv    # Pre-extracted 30-second features
└── features_3_sec.csv     # Pre-extracted 3-second chunk features (for streaming simulation)

````

**Details:**

- 1,000 audio files (100 per genre), 30 seconds each  
- Supported formats: `.wav`, `.mp3`, `.flac`, `.m4a`  
- Features extracted using **Librosa**:  
  - MFCC, chroma, spectral contrast, tonnetz, zero-crossing rate, tempo, etc.  
  - Combined into a **57-dimensional GTZAN feature vector**

---

## 🧠 Models Supported

| Model | Accuracy | Notes |
|-------|----------|-------|
| **SVM** | 71.0% | Best performing — handles high-dimensional features well |
| Random Forest | 69.5% | Interpretable and robust to overfitting |
| CNN | 10.0% | Experimental — limited dataset |

> For best accuracy and consistency, use **SVM**.  
> CNN models are optional and require TensorFlow.

---

## ⚙️ Installation

### 1️⃣ Clone Repository

```bash
git clone <repo_url>
cd <repo_folder>
````

### 2️⃣ Create and Activate Virtual Environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

**Key Dependencies:**

```
numpy, pandas, librosa, scikit-learn, matplotlib, seaborn, streamlit, joblib, tensorflow (optional)
```

---

## 🚀 Quick Start — Interactive Streamlit Dashboard

Launch the **Streamlit dashboard** for audio classification:

```bash
python run_audio_dashboard.py
```

or directly:

```bash
streamlit run app.py
```

**Features:**

* 🎵 Upload `.wav` or `.mp3` files for instant prediction
* 📈 Real-time probability visualization for each genre
* ⏱️ Simulate streaming/chunked classification
* 🔁 Compare SVM vs Random Forest model predictions
* 💾 View results and confidence scores in the dashboard

**Note:**
Make sure the following files are present before running:

```
models/
├── scaler.pkl
├── label_encoder.pkl
├── random_forest_model.pkl
├── svm_model.pkl
└── cnn_model.keras  (optional)
```

---

## 📊 Performance Summary

| Model         | Accuracy | Avg Latency     | Mode         |
| ------------- | -------- | --------------- | ------------ |
| SVM           | 71.0%    | 0.5–0.6 s/chunk | Streaming    |
| Random Forest | 69.5%    | 0.5–0.6 s/chunk | Streaming    |
| CNN           | 10.0%    | 1.2–1.5 s/chunk | Experimental |

> CNN underperforms due to limited data but demonstrates potential for future deep-learning approaches.

---

## 🧰 Troubleshooting

**Model Loading Error**

```bash
python src/model_training.py
```

**Missing Dependencies**

```bash
pip install -r requirements.txt
```

**Dashboard Not Launching**

* Ensure `streamlit` is installed
* Run from the repository root directory
* For headless environments, use saved preprocessed feature CSVs

**Performance Optimization**

* Use SSD storage for faster I/O
* Reduce chunk overlap for faster streaming
* Use SVM for best accuracy and speed trade-off

---

## 💻 Project Structure

```
src/
├── refactored_genre_classifier.py  # Main system
├── model_training.py               # Model training and saving
├── evaluation.py                   # Evaluation and metrics visualization
├── utils.py                        # Feature extraction helpers
├── example_usage.py                # Example code
└── test_refactored_system.py       # Unit tests

models/
├── scaler.pkl
├── label_encoder.pkl
├── random_forest_model.pkl
├── svm_model.pkl
└── cnn_model.keras  (optional)

Data/
├── genres_original/                 # GTZAN dataset
├── features_30_sec.csv              # 30-sec features
└── features_3_sec.csv               # 3-sec chunk features

results/                             # Output CSVs and visualizations

app.py                               # Streamlit dashboard UI
run_audio_dashboard.py               # Streamlit launcher
requirements.txt                     # Dependencies
```

---

## 🧑‍💻 Development Notes

* `models/` must contain:

  * `scaler.pkl`, `label_encoder.pkl`
  * At least one classifier (`svm_model.pkl` or `random_forest_model.pkl`)
* CNN models are optional but require TensorFlow
* Retrain models anytime using:

  ```bash
  python src/model_training.py
  ```

**To run the dashboard only, keep:**

```
app.py
run_audio_dashboard.py
models/
requirements.txt
Data/
```

---

## 🏆 Acknowledgments

* **GTZAN Dataset** creators
* **Librosa**, **Scikit-learn**, **Streamlit**, and **TensorFlow** teams
* Open-source ML and audio analysis community

---

## 📬 Contact

For any queries:
**Diya Prakash** — PES2UG23CS184
**Erin Joseph** — PES2UG23CS186
```
