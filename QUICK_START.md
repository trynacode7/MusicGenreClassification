# Quick Start Guide

## 🚀 Get Started in 5 Minutes

### 1. Setup Environment
```bash
# Create virtual environment
python -m venv venv

# Activate environment (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train Models
```bash
# Train all models (Random Forest, SVM, CNN)
python src/model_training.py
```

### 3. Run Analysis
```bash
# Generate visualizations and reports
python src/run_analysis.py
```

### 4. Real-time Prediction
```bash
# Start interactive prediction system
python src/realtime_prediction.py
```

## 📊 Expected Results

After running the analysis, you'll find in `results/`:
- `confusion_matrices.png` - Model performance matrices
- `model_accuracy_comparison.png` - Accuracy comparison chart
- `sample_spectrograms.png` - Spectrogram samples by genre
- `model_performance_summary.csv` - Detailed performance metrics

## 🎯 Performance Expectations

- **SVM Accuracy**: ~71%
- **Random Forest Accuracy**: ~69.5%
- **Real-time Latency**: ~1.3 seconds
- **Training Time**: 5-10 minutes

## 🔧 Troubleshooting

### Common Issues

1. **"Models not found"**
   ```bash
   python src/model_training.py  # Retrain models
   ```

2. **"Unicode encoding error"**
   - Use Windows Terminal or PowerShell 7+
   - Or run: `chcp 65001` in Command Prompt

3. **"Audio recording failed"**
   ```bash
   pip install sounddevice  # Install audio library
   ```

## 📁 Project Structure

```
├── Data/                    # Dataset files
├── src/                     # Source code
├── models/                  # Trained models
├── results/                 # Analysis results
├── notebooks/               # Jupyter notebooks
├── README.md               # Main documentation
├── PROJECT_REPORT.md       # Detailed report
└── QUICK_START.md          # This file
```

## 🎵 Try It Out

1. **Train Models**: `python src/model_training.py`
2. **View Results**: Check `results/` directory
3. **Real-time Demo**: `python src/realtime_prediction.py`
4. **Interactive Analysis**: `jupyter notebook notebooks/analysis.ipynb`

## 📈 What You'll Learn

- Music genre classification with machine learning
- Real-time audio processing
- Model comparison and evaluation
- Feature engineering for audio data
- Performance optimization techniques

---

**Need Help?** Check the full [README.md](README.md) or [PROJECT_REPORT.md](PROJECT_REPORT.md) for detailed documentation.
