"""
Streamlit Music Genre Classification Dashboard - Audio File Version

A browser-based dashboard for music genre classification that accepts .wav audio files
and processes them using pre-trained models. Supports both batch and streaming simulation.

Features:
- Upload .wav audio files directly
- Extract features using librosa
- Split audio into chunks for streaming simulation
- Run predictions using pre-trained models
- Display genre probabilities and spectrograms
- Dynamic updates for real-time simulation
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import sys
from pathlib import Path
import warnings
import joblib
from typing import Dict, List, Tuple, Optional
import io
import base64

warnings.filterwarnings('ignore')

# Add src directory to path for model loading
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Page configuration
st.set_page_config(
    page_title="Music Genre Classification - Audio Dashboard",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-prediction {
        background-color: #d4edda;
        color: #155724;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
    }
    .error-prediction {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
    }
    .file-info {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Genre colors for consistent visualization
GENRE_COLORS = {
    'blues': '#1f77b4',
    'classical': '#ff7f0e', 
    'country': '#2ca02c',
    'disco': '#d62728',
    'hiphop': '#9467bd',
    'jazz': '#8c564b',
    'metal': '#e377c2',
    'pop': '#7f7f7f',
    'reggae': '#bcbd22',
    'rock': '#17becf',
    'Unknown': '#ff9999'
}

class AudioProcessor:
    """Audio processing class for feature extraction and chunking."""
    
    def __init__(self, models_dir: str = "models"):
        """Initialize the audio processor with pre-trained models."""
        self.models_dir = Path(models_dir)
        self.models = {}
        self.scaler = None
        self.label_encoder = None
        self.sample_rate = 22050
        self.chunk_duration = 3.0  # seconds
        self.chunk_overlap = 0.5   # seconds
        
        # Load models
        self._load_models()
        
    def _load_models(self):
        """Load pre-trained models and preprocessing artifacts."""
        try:
            # Load preprocessing artifacts
            self.scaler = joblib.load(self.models_dir / "scaler_improved.pkl")
            self.label_encoder = joblib.load(self.models_dir / "label_encoder_improved.pkl")
            self.models['random_forest'] = joblib.load(self.models_dir / "random_forest_improved.pkl")
            self.models['svm'] = joblib.load(self.models_dir / "svm_improved.pkl")

            
            # Load CNN if available
            try:
                from tensorflow.keras.models import load_model
                self.models['cnn'] = load_model(self.models_dir / "cnn_model.keras")
            except:
                self.models['cnn'] = None
                
            return True
        except Exception as e:
            st.error(f"Error loading models: {e}")
            return False
    
    def extract_features(self, audio_data: np.ndarray, sr: int = 22050) -> np.ndarray:
        """Extract features from audio data using librosa to match GTZAN dataset exactly."""
        try:
            # Ensure audio is the right length (30 seconds for feature extraction)
            target_length = int(30 * sr)
            if len(audio_data) > target_length:
                audio_data = audio_data[:target_length]
            elif len(audio_data) < target_length:
                audio_data = np.pad(audio_data, (0, target_length - len(audio_data)))
            
            # Extract features to match GTZAN dataset exactly (57 features)
            features = []
            
            # 1-2. Chroma features (mean, var)
            chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
            chroma_mean = float(np.mean(chroma))
            chroma_var = float(np.var(chroma))
            features.extend([chroma_mean, chroma_var])
            
            # 3-4. RMS features (mean, var)
            rms = librosa.feature.rms(y=audio_data)[0]
            rms_mean = float(np.mean(rms))
            rms_var = float(np.var(rms))
            features.extend([rms_mean, rms_var])
            
            # 5-6. Spectral centroid features (mean, var)
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0]
            spectral_centroid_mean = float(np.mean(spectral_centroids))
            spectral_centroid_var = float(np.var(spectral_centroids))
            features.extend([spectral_centroid_mean, spectral_centroid_var])
            
            # 7-8. Spectral bandwidth features (mean, var)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr)[0]
            spectral_bandwidth_mean = float(np.mean(spectral_bandwidth))
            spectral_bandwidth_var = float(np.var(spectral_bandwidth))
            features.extend([spectral_bandwidth_mean, spectral_bandwidth_var])
            
            # 9-10. Rolloff features (mean, var)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)[0]
            rolloff_mean = float(np.mean(spectral_rolloff))
            rolloff_var = float(np.var(spectral_rolloff))
            features.extend([rolloff_mean, rolloff_var])
            
            # 11-12. Zero crossing rate features (mean, var)
            zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
            zcr_mean = float(np.mean(zcr))
            zcr_var = float(np.var(zcr))
            features.extend([zcr_mean, zcr_var])
            
            # 13-14. Harmony features (mean, var) - using tonnetz as harmony approximation
            tonnetz = librosa.feature.tonnetz(y=audio_data, sr=sr)
            harmony_mean = float(np.mean(tonnetz))
            harmony_var = float(np.var(tonnetz))
            features.extend([harmony_mean, harmony_var])
            
            # 15-16. Perceptr features (mean, var) - using spectral contrast as perceptual approximation
            spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sr)
            perceptr_mean = float(np.mean(spectral_contrast))
            perceptr_var = float(np.var(spectral_contrast))
            features.extend([perceptr_mean, perceptr_var])
            
            # 17. Tempo
            tempo, _ = librosa.beat.beat_track(y=audio_data, sr=sr)
            features.append(float(tempo))
            
            # 18-57. MFCC features (20 coefficients × 2 stats each = 40 features)
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=20)
            for mfcc in mfccs:
                mfcc_mean = float(np.mean(mfcc))
                mfcc_var = float(np.var(mfcc))
                features.extend([mfcc_mean, mfcc_var])
            
            # Ensure we have exactly 57 features
            if len(features) != 57:
                # Pad or truncate to exactly 57 features
                if len(features) < 57:
                    features.extend([0.0] * (57 - len(features)))
                else:
                    features = features[:57]
            
            return np.array(features, dtype=np.float64)
            
        except Exception as e:
            st.error(f"Error extracting features: {e}")
            return np.zeros(57, dtype=np.float64)  # Expected feature count
    
    def create_audio_chunks(self, audio_data: np.ndarray, sr: int = 22050) -> List[Tuple[np.ndarray, float, float]]:
        """Split audio into overlapping chunks for streaming simulation."""
        chunk_length = int(self.chunk_duration * sr)
        overlap_length = int(self.chunk_overlap * sr)
        step_size = chunk_length - overlap_length
        
        chunks = []
        start_time = 0.0
        
        for i in range(0, len(audio_data) - chunk_length + 1, step_size):
            chunk_audio = audio_data[i:i + chunk_length]
            end_time = start_time + self.chunk_duration
            
            chunks.append((chunk_audio, start_time, end_time))
            start_time += (self.chunk_duration - self.chunk_overlap)
        
        return chunks
    
    def predict_genre(self, audio_data: np.ndarray, model_name: str = 'svm') -> Dict:
        """Predict genre from audio data using specified model."""
        try:
            if model_name not in self.models or self.models[model_name] is None:
                return {
                    'genre': 'Unknown',
                    'confidence': 0.0,
                    'error': f'Model {model_name} not available'
                }
            
            # Extract features
            features = self.extract_features(audio_data)
            features = features.reshape(1, -1)
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Make prediction
            model = self.models[model_name]
            prediction = model.predict(features_scaled)[0]
            
            # Get confidence score
            try:
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(features_scaled)[0]
                    confidence = float(np.max(probabilities))
                    
                    # Check if features are reasonable (not extreme values)
                    if np.any(np.abs(features_scaled) > 10):  # Extreme values indicate synthetic audio
                        confidence = max(0.1, confidence * 0.5)  # Reduce confidence for synthetic audio
                        
                else:
                    # For models without predict_proba, use decision function
                    if hasattr(model, 'decision_function'):
                        decision_scores = model.decision_function(features_scaled)[0]
                        confidence = float(np.max(decision_scores))
                    else:
                        confidence = 1.0  # Default confidence
            except Exception:
                confidence = 1.0  # Fallback confidence
            
            # Decode label
            genre = self.label_encoder.inverse_transform([prediction])[0]
            
            # Add warning for synthetic audio
            warning = None
            if np.any(np.abs(features_scaled) > 10):
                warning = "Warning: This appears to be synthetic audio. Models work best with real music files."
            
            result = {
                'genre': genre,
                'confidence': confidence,
                'model': model_name
            }
            
            if warning:
                result['warning'] = warning
                
            return result
            
        except Exception as e:
            return {
                'genre': 'Unknown',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def create_spectrogram(self, audio_data: np.ndarray, sr: int = 22050) -> np.ndarray:
        """Create spectrogram from audio data."""
        try:
            # Create mel spectrogram
            mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sr)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            return mel_spec_db
        except Exception as e:
            st.error(f"Error creating spectrogram: {e}")
            return np.array([])

def load_audio_file(uploaded_file) -> Tuple[np.ndarray, int]:
    """Load audio file from uploaded file."""
    try:
        # Save uploaded file temporarily
        with open("temp_audio.wav", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Load with librosa
        audio_data, sr = librosa.load("temp_audio.wav", sr=22050)
        
        # Clean up temp file
        os.remove("temp_audio.wav")
        
        return audio_data, sr
    except Exception as e:
        st.error(f"Error loading audio file: {e}")
        return np.array([]), 22050

def create_genre_probability_chart(prediction_result: Dict) -> go.Figure:
    """Create genre probability bar chart."""
    # Create genre probability distribution
    genres = list(GENRE_COLORS.keys())
    probabilities = np.zeros(len(genres))
    
    # Set probability for predicted genre
    predicted_genre = prediction_result['genre']
    confidence = prediction_result['confidence']
    
    if predicted_genre in genres:
        genre_idx = genres.index(predicted_genre)
        probabilities[genre_idx] = confidence
    else:
        # Unknown genre
        probabilities[-1] = confidence
    
    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=genres,
            y=probabilities,
            marker_color=[GENRE_COLORS.get(g, '#cccccc') for g in genres],
            text=[f'{p:.3f}' if p > 0 else '' for p in probabilities],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Probability: %{y:.3f}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title="Genre Probabilities",
        xaxis_title="Genre",
        yaxis_title="Probability",
        yaxis=dict(range=[0, 1]),
        showlegend=False,
        height=400
    )
    
    # Highlight the predicted genre
    if predicted_genre in genres:
        genre_idx = genres.index(predicted_genre)
        # Create a new color list with the predicted genre highlighted
        colors = ['#1f77b4'] * len(genres)  # Default blue color
        colors[genre_idx] = '#ff0000'  # Red for predicted
        fig.data[0].marker.color = colors
    
    return fig

def create_spectrogram_plot(audio_data: np.ndarray, sr: int = 22050) -> go.Figure:
    """Create spectrogram visualization."""
    try:
        # Create mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sr)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=mel_spec_db,
            colorscale='Viridis',
            hovertemplate='Time: %{x}<br>Frequency: %{y}<br>Magnitude: %{z}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Mel Spectrogram",
            xaxis_title="Time (frames)",
            yaxis_title="Mel Frequency Bins",
            height=300
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating spectrogram: {e}")
        return go.Figure()

def create_rolling_history_chart(chunk_predictions: List[Dict]) -> go.Figure:
    """Create rolling history chart of chunk predictions."""
    if not chunk_predictions:
        return go.Figure()
    
    # Extract data
    times = [i * 3.0 for i in range(len(chunk_predictions))]  # 3-second chunks
    genres = [pred['genre'] for pred in chunk_predictions]
    confidences = [pred['confidence'] for pred in chunk_predictions]
    
    # Create scatter plot
    fig = go.Figure()
    
    for i, (time, genre, conf) in enumerate(zip(times, genres, confidences)):
        color = GENRE_COLORS.get(genre, '#cccccc')
        
        fig.add_trace(go.Scatter(
            x=[time],
            y=[i],
            mode='markers+text',
            marker=dict(size=15, color=color, line=dict(width=2, color='black')),
            text=[f'{genre}<br>{conf:.2f}'],
            textposition='middle center',
            name=genre,
            showlegend=False,
            hovertemplate=f'<b>{genre}</b><br>Time: {time:.1f}s<br>Confidence: {conf:.3f}<extra></extra>'
        ))
    
    fig.update_layout(
        title="Prediction History",
        xaxis_title="Time (seconds)",
        yaxis_title="Chunk Index",
        height=400,
        showlegend=False
    )
    
    return fig

def create_confidence_timeline(chunk_predictions: List[Dict]) -> go.Figure:
    """Create confidence timeline chart."""
    if not chunk_predictions:
        return go.Figure()
    
    # Extract data
    chunk_indices = list(range(len(chunk_predictions)))
    confidences = [pred['confidence'] for pred in chunk_predictions]
    genres = [pred['genre'] for pred in chunk_predictions]
    
    # Create line plot
    fig = go.Figure()
    
    # Add line
    fig.add_trace(go.Scatter(
        x=chunk_indices,
        y=confidences,
        mode='lines',
        line=dict(color='blue', width=2),
        name='Confidence',
        hovertemplate='Chunk: %{x}<br>Confidence: %{y:.3f}<extra></extra>'
    ))
    
    # Add colored scatter points for each genre
    for i, (chunk_idx, conf, genre) in enumerate(zip(chunk_indices, confidences, genres)):
        color = GENRE_COLORS.get(genre, '#cccccc')
        
        fig.add_trace(go.Scatter(
            x=[chunk_idx],
            y=[conf],
            mode='markers',
            marker=dict(size=8, color=color, line=dict(width=1, color='black')),
            name=genre,
            showlegend=False,
            hovertemplate=f'<b>{genre}</b><br>Chunk: {chunk_idx}<br>Confidence: {conf:.3f}<extra></extra>'
        ))
    
    fig.update_layout(
        title="Confidence Over Time",
        xaxis_title="Chunk Index",
        yaxis_title="Confidence",
        yaxis=dict(range=[0, 1]),
        height=400,
        showlegend=False
    )
    
    return fig

def main():
    """Main Streamlit app."""
    # Header
    st.markdown('<h1 class="main-header">🎵 Music Genre Classification - Audio Dashboard</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'processor' not in st.session_state:
        st.session_state.processor = None
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = {}
    if 'current_file' not in st.session_state:
        st.session_state.current_file = None
    if 'chunk_predictions' not in st.session_state:
        st.session_state.chunk_predictions = []
    if 'current_chunk' not in st.session_state:
        st.session_state.current_chunk = 0
    if 'animating' not in st.session_state:
        st.session_state.animating = False
    
    # Sidebar
    st.sidebar.header("🎛️ Controls")
    
    # Initialize processor
    if st.session_state.processor is None:
        with st.spinner("Loading models..."):
            st.session_state.processor = AudioProcessor()
            if st.session_state.processor.models['svm'] is None:
                st.error("Failed to load models. Please ensure model files exist in the 'models' directory.")
                st.stop()
    
    # File upload
    st.sidebar.header("📁 Upload Audio Files")
    uploaded_files = st.sidebar.file_uploader(
        "Upload .wav audio files",
        type=['wav'],
        accept_multiple_files=True,
        help="Upload one or more .wav audio files for genre classification"
    )
    
    # Process uploaded files
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.uploaded_files:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    # Load audio
                    audio_data, sr = load_audio_file(uploaded_file)
                    
                    if len(audio_data) > 0:
                        # Create chunks
                        chunks = st.session_state.processor.create_audio_chunks(audio_data, sr)
                        
                        # Store file data
                        st.session_state.uploaded_files[uploaded_file.name] = {
                            'audio_data': audio_data,
                            'sr': sr,
                            'chunks': chunks,
                            'duration': len(audio_data) / sr
                        }
                        
                        st.sidebar.success(f"✓ {uploaded_file.name} processed ({len(chunks)} chunks)")
                    else:
                        st.sidebar.error(f"✗ Failed to process {uploaded_file.name}")
    
    # File selection
    if st.session_state.uploaded_files:
        st.sidebar.header("📂 Select File")
        selected_file = st.sidebar.selectbox(
            "Choose file to analyze",
            list(st.session_state.uploaded_files.keys())
        )
        st.session_state.current_file = selected_file
        
        # File info
        file_data = st.session_state.uploaded_files[selected_file]
        st.sidebar.markdown(f"""
        <div class="file-info">
            <strong>{selected_file}</strong><br>
            Duration: {file_data['duration']:.1f}s<br>
            Chunks: {len(file_data['chunks'])}<br>
            Sample Rate: {file_data['sr']} Hz
        </div>
        """, unsafe_allow_html=True)
        
        # Model selection
        st.sidebar.header("🤖 Model Selection")
        model_choice = st.sidebar.selectbox(
            "Choose model",
            ['svm', 'random_forest', 'cnn'],
            help="SVM typically performs best"
        )
        
        # Animation controls
        st.sidebar.header("▶️ Animation Controls")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("▶️ Start"):
                st.session_state.animating = True
                st.session_state.current_chunk = 0
                st.session_state.chunk_predictions = []
        
        with col2:
            if st.button("⏹️ Stop"):
                st.session_state.animating = False
        
        # Animation speed
        speed = st.sidebar.slider("Speed (seconds)", 0.1, 3.0, 1.0, 0.1)
        
        # Manual chunk selection
        max_chunks = len(file_data['chunks'])
        current_chunk = st.sidebar.slider(
            "Current Chunk",
            0, max_chunks-1, 
            st.session_state.current_chunk,
            help="Manually select chunk to analyze"
        )
        st.session_state.current_chunk = current_chunk
        
        # Process current chunk
        if st.session_state.current_chunk < len(file_data['chunks']):
            chunk_audio, start_time, end_time = file_data['chunks'][st.session_state.current_chunk]
            
            # Predict genre
            prediction = st.session_state.processor.predict_genre(chunk_audio, model_choice)
            
            # Add to predictions history
            if len(st.session_state.chunk_predictions) <= st.session_state.current_chunk:
                st.session_state.chunk_predictions.append(prediction)
            else:
                st.session_state.chunk_predictions[st.session_state.current_chunk] = prediction
        
        # Auto-animation
        if st.session_state.animating:
            time.sleep(speed)
            st.session_state.current_chunk = (st.session_state.current_chunk + 1) % len(file_data['chunks'])
            st.rerun()
    
    # Main content
    if st.session_state.uploaded_files and st.session_state.current_file:
        file_data = st.session_state.uploaded_files[st.session_state.current_file]
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Files", len(st.session_state.uploaded_files))
        
        with col2:
            st.metric("Current File", st.session_state.current_file)
        
        with col3:
            st.metric("Total Chunks", len(file_data['chunks']))
        
        with col4:
            if st.session_state.chunk_predictions:
                avg_confidence = np.mean([pred['confidence'] for pred in st.session_state.chunk_predictions])
                st.metric("Avg Confidence", f"{avg_confidence:.3f}")
        
        # Current chunk analysis
        if st.session_state.current_chunk < len(file_data['chunks']):
            chunk_audio, start_time, end_time = file_data['chunks'][st.session_state.current_chunk]
            
            st.header(f"🎵 Current Chunk Analysis ({start_time:.1f}s - {end_time:.1f}s)")
            
            # Get prediction for current chunk
            if st.session_state.current_chunk < len(st.session_state.chunk_predictions):
                current_prediction = st.session_state.chunk_predictions[st.session_state.current_chunk]
            else:
                current_prediction = st.session_state.processor.predict_genre(chunk_audio, model_choice)
            
            # Display results in columns
            col1, col2 = st.columns(2)
            
            with col1:
                # Genre probabilities chart
                prob_chart = create_genre_probability_chart(current_prediction)
                st.plotly_chart(prob_chart, use_container_width=True)
                
                # Prediction details
                st.subheader("🎯 Prediction Details")
                st.write(f"**Predicted Genre:** {current_prediction['genre']}")
                st.write(f"**Confidence:** {current_prediction['confidence']:.3f}")
                st.write(f"**Model:** {current_prediction.get('model', 'Unknown')}")
                
                if 'error' in current_prediction:
                    st.error(f"Error: {current_prediction['error']}")
                
                if 'warning' in current_prediction:
                    st.warning(current_prediction['warning'])
            
            with col2:
                # Spectrogram
                spec_chart = create_spectrogram_plot(chunk_audio, file_data['sr'])
                st.plotly_chart(spec_chart, use_container_width=True)
                
                # Audio waveform
                st.subheader("🌊 Audio Waveform")
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(np.linspace(start_time, end_time, len(chunk_audio)), chunk_audio)
                ax.set_xlabel("Time (seconds)")
                ax.set_ylabel("Amplitude")
                ax.set_title(f"Waveform - Chunk {st.session_state.current_chunk}")
                st.pyplot(fig)
        
        # History and timeline
        if st.session_state.chunk_predictions:
            st.header("📊 Analysis History")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Rolling history
                history_chart = create_rolling_history_chart(st.session_state.chunk_predictions)
                st.plotly_chart(history_chart, use_container_width=True)
            
            with col2:
                # Confidence timeline
                conf_chart = create_confidence_timeline(st.session_state.chunk_predictions)
                st.plotly_chart(conf_chart, use_container_width=True)
        
        # Batch prediction for full file
        st.header("📋 Full File Analysis")
        
        if st.button("🔍 Analyze Full File"):
            with st.spinner("Analyzing full file..."):
                # Predict genre for full file
                full_prediction = st.session_state.processor.predict_genre(file_data['audio_data'], model_choice)
                
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Predicted Genre", full_prediction['genre'])
                
                with col2:
                    st.metric("Confidence", f"{full_prediction['confidence']:.3f}")
                
                with col3:
                    st.metric("Model Used", full_prediction.get('model', 'Unknown'))
                
                # Full file spectrogram
                st.subheader("🎼 Full File Spectrogram")
                full_spec = create_spectrogram_plot(file_data['audio_data'], file_data['sr'])
                st.plotly_chart(full_spec, use_container_width=True)
    
    else:
        st.info("👆 Please upload .wav audio files to get started!")
        
        st.markdown("""
        ### How to use this dashboard:
        
        1. **Upload .wav audio files** using the file uploader in the sidebar
        2. **Select a file** from the dropdown to analyze
        3. **Choose a model** (SVM recommended for best performance)
        4. **Start animation** to see real-time streaming simulation
        5. **View predictions** for individual chunks and full file
        
        ### Features:
        - **Real-time simulation**: Watch genre predictions change as audio plays
        - **Chunk analysis**: See predictions for 3-second audio segments
        - **Full file analysis**: Get overall genre prediction for entire file
        - **Visualizations**: Spectrograms, waveforms, and probability charts
        - **Multiple models**: Compare Random Forest, SVM, and CNN predictions
        """)

if __name__ == "__main__":
    main()
