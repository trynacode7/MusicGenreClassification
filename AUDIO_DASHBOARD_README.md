# Music Genre Classification - Audio Dashboard

A comprehensive Streamlit dashboard for music genre classification that accepts .wav audio files directly and processes them using pre-trained machine learning models.

## 🎵 Features

### **Audio File Processing**
- Upload .wav audio files directly through the web interface
- Automatic feature extraction using librosa
- Audio chunking for streaming simulation (3-second chunks with 0.5s overlap)
- Support for multiple audio files

### **Real-Time Streaming Simulation**
- Dynamic genre probability visualization
- Rolling history of chunk predictions
- Confidence timeline charts
- Animated updates to simulate live streaming

### **Visualizations**
- **Genre Probability Bar Charts**: Show probability distribution across all genres
- **Spectrograms**: Mel-frequency spectrograms for audio analysis
- **Waveforms**: Time-domain audio visualization
- **Prediction History**: Timeline of genre predictions
- **Confidence Charts**: Confidence scores over time

### **Multiple Model Support**
- **SVM**: Best performing model (71% accuracy)
- **Random Forest**: Robust ensemble method (69.5% accuracy)
- **CNN**: Deep learning approach (experimental)

### **Batch and Streaming Analysis**
- **Chunk Analysis**: Individual predictions for 3-second segments
- **Full File Analysis**: Overall genre prediction for entire audio file
- **Model Comparison**: Switch between different models

## 🚀 Quick Start

### **1. Install Dependencies**
```bash
pip install -r requirements_audio_dashboard.txt
```

### **2. Ensure Models are Trained**
```bash
python src/model_training.py
```

### **3. Launch Dashboard**
```bash
python run_audio_dashboard.py
```

Or directly:
```bash
streamlit run src/streamlit_audio_dashboard.py
```

### **4. Open Browser**
Navigate to: http://localhost:8501

## 📁 File Structure

```
src/
├── streamlit_audio_dashboard.py    # Main audio dashboard
├── refactored_genre_classifier.py  # Core classification system
├── model_training.py              # Model training script
└── utils.py                       # Utility functions

models/
├── scaler.pkl                     # Feature scaler
├── label_encoder.pkl             # Label encoder
├── svm_model.pkl                 # SVM model
├── random_forest_model.pkl       # Random Forest model
└── cnn_model.keras               # CNN model (optional)

requirements_audio_dashboard.txt   # Dashboard dependencies
run_audio_dashboard.py            # Easy launcher
AUDIO_DASHBOARD_README.md         # This documentation
```

## 🎛️ Dashboard Usage

### **1. Upload Audio Files**
- Use the file uploader in the sidebar
- Supports multiple .wav files
- Files are automatically processed and chunked

### **2. Select File to Analyze**
- Choose from uploaded files using the dropdown
- View file information (duration, chunks, sample rate)

### **3. Choose Model**
- Select from SVM, Random Forest, or CNN
- SVM is recommended for best performance

### **4. Streaming Simulation**
- **Start Animation**: Begin real-time simulation
- **Speed Control**: Adjust animation speed (0.1-3.0 seconds)
- **Manual Navigation**: Use slider to select specific chunks

### **5. View Results**
- **Current Chunk**: Genre probabilities and spectrogram
- **Prediction History**: Timeline of chunk predictions
- **Full File Analysis**: Overall genre prediction

## 📊 Dashboard Components

### **Sidebar Controls**
- **File Upload**: Upload .wav audio files
- **File Selection**: Choose which file to analyze
- **Model Selection**: Choose classification model
- **Animation Controls**: Start/stop streaming simulation
- **Speed Control**: Adjust animation speed
- **Chunk Navigation**: Manual chunk selection

### **Main Dashboard**
- **Metrics Row**: File count, current file, chunks, confidence
- **Current Chunk Analysis**: Genre probabilities, spectrogram, waveform
- **Analysis History**: Rolling history and confidence timeline
- **Full File Analysis**: Complete file genre prediction

### **Visualizations**
- **Genre Probability Bar Chart**: Real-time genre probabilities
- **Mel Spectrogram**: Frequency-time representation
- **Audio Waveform**: Time-domain audio visualization
- **Prediction History**: Timeline of genre predictions
- **Confidence Timeline**: Confidence scores over time

## 🔧 Technical Details

### **Audio Processing**
- **Sample Rate**: 22050 Hz (standard for librosa)
- **Chunk Duration**: 3 seconds
- **Chunk Overlap**: 0.5 seconds
- **Feature Extraction**: 58-dimensional feature vector

### **Feature Extraction**
- **Spectral Features**: Centroid, rolloff, bandwidth
- **MFCC Features**: 13 Mel-frequency cepstral coefficients
- **Chroma Features**: 12 chroma features
- **Rhythm Features**: Tempo estimation
- **Energy Features**: RMS energy

### **Model Performance**
- **SVM**: 71.0% accuracy, 71.0% F1-score
- **Random Forest**: 69.5% accuracy, 69.0% F1-score
- **CNN**: 10.0% accuracy (experimental)

## 🎯 Use Cases

### **Music Analysis**
- Analyze individual songs for genre classification
- Compare predictions across different models
- Visualize audio characteristics through spectrograms

### **Streaming Simulation**
- Simulate real-time genre classification
- Test model performance on audio chunks
- Analyze confidence patterns over time

### **Research and Development**
- Compare model performance on different audio files
- Analyze feature importance through visualizations
- Test model robustness across various audio types

## 🛠️ Troubleshooting

### **Common Issues**

1. **Models Not Loading**
   ```bash
   # Retrain models
   python src/model_training.py
   ```

2. **Missing Dependencies**
   ```bash
   # Install requirements
   pip install -r requirements_audio_dashboard.txt
   ```

3. **Audio Loading Errors**
   - Ensure files are in .wav format
   - Check file is not corrupted
   - Verify file permissions

4. **Dashboard Not Loading**
   - Check if port 8501 is available
   - Try different port: `streamlit run src/streamlit_audio_dashboard.py --server.port 8502`

### **Performance Optimization**

1. **For Large Files**
   - Reduce chunk overlap for faster processing
   - Use SVM model for best speed/accuracy trade-off

2. **For Real-Time Simulation**
   - Adjust animation speed for smooth playback
   - Use smaller audio files for better responsiveness

## 📈 Advanced Features

### **Custom Model Integration**
```python
# Add your own model to the processor
processor.models['custom_model'] = your_model
```

### **Custom Feature Extraction**
```python
# Override feature extraction method
def custom_extract_features(audio_data, sr):
    # Your custom feature extraction
    return features
```

### **Batch Processing**
```python
# Process multiple files programmatically
for file_path in audio_files:
    audio_data, sr = librosa.load(file_path)
    prediction = processor.predict_genre(audio_data)
```

## 🎉 Success Metrics

- ✅ **Direct Audio Upload**: No CSV files required
- ✅ **Real-Time Simulation**: Streaming-like experience
- ✅ **Multiple Visualizations**: Comprehensive audio analysis
- ✅ **Model Comparison**: Easy switching between models
- ✅ **Professional UI**: Clean, intuitive interface
- ✅ **Modular Design**: Easy to extend and customize

## 🚀 Next Steps

1. **Upload your .wav files** to the dashboard
2. **Start with SVM model** for best performance
3. **Use streaming simulation** to see real-time predictions
4. **Analyze full files** for overall genre classification
5. **Compare models** to see performance differences

The audio dashboard provides a complete solution for music genre classification with an intuitive web interface, real-time visualization, and comprehensive analysis capabilities.

---

**🎵 Music Genre Classification - Audio Dashboard Ready! 🎵**
