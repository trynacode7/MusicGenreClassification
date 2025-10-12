"""
Real-time music genre prediction system.

This module implements near-real-time genre prediction from microphone input
or short audio clips using trained ML models.
"""

import numpy as np
import pandas as pd
import librosa
import sounddevice as sd
import time
import threading
from typing import Dict, List, Tuple, Optional
import joblib
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import our trained models
try:
    from tensorflow.keras.models import load_model
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not available. CNN predictions will be disabled.")

class RealTimePredictor:
    """
    Real-time music genre prediction system.
    
    Supports prediction using Random Forest, SVM, and CNN models.
    Can process live audio from microphone or pre-recorded audio files.
    """
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize the real-time predictor.
        
        Args:
            models_dir: Directory containing trained models
        """
        self.models_dir = Path(models_dir)
        self.models = {}
        self.scaler = None
        self.label_encoder = None
        self.is_recording = False
        self.audio_buffer = []
        self.sample_rate = 22050  # Standard sample rate for librosa
        self.chunk_duration = 3.0  # Duration of audio chunks in seconds
        
        # Load models and preprocessing artifacts
        self._load_models()
        
    def _load_models(self):
        """Load all trained models and preprocessing artifacts."""
        try:
            # Load preprocessing artifacts
            self.scaler = joblib.load(self.models_dir / "scaler.pkl")
            print("[OK] Scaler loaded successfully")
            
            self.label_encoder = joblib.load(self.models_dir / "label_encoder.pkl")
            print("[OK] Label encoder loaded successfully")
            
            # Load Random Forest model
            self.models['random_forest'] = joblib.load(self.models_dir / "random_forest_model.pkl")
            print("[OK] Random Forest model loaded successfully")
            
            # Load SVM model
            self.models['svm'] = joblib.load(self.models_dir / "svm_model.pkl")
            print("[OK] SVM model loaded successfully")
            
            # Load CNN model if available
            if TENSORFLOW_AVAILABLE:
                try:
                    self.models['cnn'] = load_model(self.models_dir / "cnn_model.h5")
                    print("[OK] CNN model loaded successfully")
                except Exception as e:
                    print(f"Warning: Could not load CNN model: {e}")
                    self.models['cnn'] = None
            else:
                self.models['cnn'] = None
                print("Warning: TensorFlow not available, CNN predictions disabled")
                
        except Exception as e:
            print(f"Error loading models: {e}")
            raise
    
    def extract_features(self, audio_data: np.ndarray, sr: int = 22050) -> np.ndarray:
        """
        Extract features from audio data using librosa.
        
        Args:
            audio_data: Audio signal
            sr: Sample rate
            
        Returns:
            Feature vector
        """
        try:
            # Ensure audio is the right length (30 seconds for feature extraction)
            target_length = int(30 * sr)
            if len(audio_data) > target_length:
                audio_data = audio_data[:target_length]
            elif len(audio_data) < target_length:
                audio_data = np.pad(audio_data, (0, target_length - len(audio_data)))
            
            # Extract features similar to GTZAN dataset
            features = []
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0]
            features.extend([
                np.mean(spectral_centroids),
                np.std(spectral_centroids),
                np.min(spectral_centroids),
                np.max(spectral_centroids)
            ])
            
            # Spectral rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)[0]
            features.extend([
                np.mean(spectral_rolloff),
                np.std(spectral_rolloff)
            ])
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
            features.extend([
                np.mean(zcr),
                np.std(zcr)
            ])
            
            # MFCC features (13 coefficients)
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
            for mfcc in mfccs:
                features.extend([
                    np.mean(mfcc),
                    np.std(mfcc)
                ])
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
            for chroma_vector in chroma:
                features.extend([
                    np.mean(chroma_vector),
                    np.std(chroma_vector)
                ])
            
            # Tempo
            tempo, _ = librosa.beat.beat_track(y=audio_data, sr=sr)
            features.append(tempo)
            
            # Spectral bandwidth
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr)[0]
            features.extend([
                np.mean(spectral_bandwidth),
                np.std(spectral_bandwidth)
            ])
            
            # RMS energy
            rms = librosa.feature.rms(y=audio_data)[0]
            features.extend([
                np.mean(rms),
                np.std(rms)
            ])
            
            return np.array(features)
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            # Return zero features if extraction fails
            return np.zeros(58)  # Expected feature count
    
    def predict_genre(self, audio_data: np.ndarray, model_name: str = 'random_forest') -> Dict:
        """
        Predict genre from audio data using specified model.
        
        Args:
            audio_data: Audio signal
            model_name: Name of model to use ('random_forest', 'svm', 'cnn')
            
        Returns:
            Dictionary with prediction results
        """
        start_time = time.time()
        
        try:
            if model_name not in self.models or self.models[model_name] is None:
                return {
                    'genre': 'Unknown',
                    'confidence': 0.0,
                    'error': f'Model {model_name} not available',
                    'latency': 0.0
                }
            
            if model_name == 'cnn':
                # CNN prediction (requires spectrogram)
                return self._predict_cnn(audio_data)
            else:
                # Traditional ML prediction (Random Forest, SVM)
                return self._predict_traditional_ml(audio_data, model_name)
                
        except Exception as e:
            return {
                'genre': 'Unknown',
                'confidence': 0.0,
                'error': str(e),
                'latency': time.time() - start_time
            }
    
    def _predict_traditional_ml(self, audio_data: np.ndarray, model_name: str) -> Dict:
        """Predict using traditional ML models (Random Forest, SVM)."""
        start_time = time.time()
        
        # Extract features
        features = self.extract_features(audio_data)
        features = features.reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Make prediction
        model = self.models[model_name]
        prediction = model.predict(features_scaled)[0]
        confidence = model.predict_proba(features_scaled)[0].max()
        
        # Decode label
        genre = self.label_encoder.inverse_transform([prediction])[0]
        
        latency = time.time() - start_time
        
        return {
            'genre': genre,
            'confidence': confidence,
            'latency': latency,
            'model': model_name
        }
    
    def _predict_cnn(self, audio_data: np.ndarray) -> Dict:
        """Predict using CNN model (requires spectrogram)."""
        start_time = time.time()
        
        try:
            # Generate spectrogram
            stft = librosa.stft(audio_data)
            spectrogram = np.abs(stft)
            
            # Convert to mel spectrogram
            mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=self.sample_rate)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Resize to match training size (128x128)
            from scipy.ndimage import zoom
            target_size = (128, 128)
            zoom_factors = (target_size[0] / mel_spec_db.shape[0], 
                          target_size[1] / mel_spec_db.shape[1])
            mel_spec_resized = zoom(mel_spec_db, zoom_factors)
            
            # Normalize
            mel_spec_resized = (mel_spec_resized - mel_spec_resized.min()) / (mel_spec_resized.max() - mel_spec_resized.min())
            
            # Add batch and channel dimensions
            mel_spec_input = mel_spec_resized.reshape(1, 128, 128, 1)
            
            # Make prediction
            predictions = self.models['cnn'].predict(mel_spec_input, verbose=0)
            prediction = np.argmax(predictions[0])
            confidence = predictions[0].max()
            
            # Decode label
            genre = self.label_encoder.inverse_transform([prediction])[0]
            
            latency = time.time() - start_time
            
            return {
                'genre': genre,
                'confidence': confidence,
                'latency': latency,
                'model': 'cnn'
            }
            
        except Exception as e:
            return {
                'genre': 'Unknown',
                'confidence': 0.0,
                'error': str(e),
                'latency': time.time() - start_time
            }
    
    def record_audio(self, duration: float = 3.0) -> np.ndarray:
        """
        Record audio from microphone.
        
        Args:
            duration: Recording duration in seconds
            
        Returns:
            Recorded audio data
        """
        print(f"Recording for {duration} seconds...")
        print("Speak/sing now!")
        
        # Record audio
        audio_data = sd.rec(int(duration * self.sample_rate), 
                          samplerate=self.sample_rate, 
                          channels=1, 
                          dtype='float64')
        sd.wait()  # Wait until recording is finished
        
        print("Recording finished!")
        return audio_data.flatten()
    
    def predict_from_file(self, file_path: str, model_name: str = 'random_forest') -> Dict:
        """
        Predict genre from audio file.
        
        Args:
            file_path: Path to audio file
            model_name: Model to use for prediction
            
        Returns:
            Prediction results
        """
        try:
            # Load audio file
            audio_data, sr = librosa.load(file_path, sr=self.sample_rate)
            
            # Predict genre
            result = self.predict_genre(audio_data, model_name)
            result['file_path'] = file_path
            
            return result
            
        except Exception as e:
            return {
                'genre': 'Unknown',
                'confidence': 0.0,
                'error': str(e),
                'file_path': file_path,
                'latency': 0.0
            }
    
    def compare_models(self, audio_data: np.ndarray) -> Dict:
        """
        Compare predictions from all available models.
        
        Args:
            audio_data: Audio signal
            
        Returns:
            Dictionary with predictions from all models
        """
        results = {}
        
        for model_name in ['random_forest', 'svm', 'cnn']:
            if self.models[model_name] is not None:
                results[model_name] = self.predict_genre(audio_data, model_name)
        
        return results
    
    def real_time_prediction_demo(self, duration: float = 10.0):
        """
        Demo real-time prediction with continuous recording.
        
        Args:
            duration: Total demo duration in seconds
        """
        print(f"Starting real-time prediction demo for {duration} seconds...")
        print("Press Ctrl+C to stop early")
        
        start_time = time.time()
        
        try:
            while time.time() - start_time < duration:
                # Record short audio chunk
                audio_chunk = self.record_audio(self.chunk_duration)
                
                # Get predictions from all models
                predictions = self.compare_models(audio_chunk)
                
                # Display results
                print(f"\n--- Prediction Results (t={time.time()-start_time:.1f}s) ---")
                for model_name, result in predictions.items():
                    if 'error' not in result:
                        print(f"{model_name.upper()}: {result['genre']} "
                              f"(confidence: {result['confidence']:.3f}, "
                              f"latency: {result['latency']:.3f}s)")
                    else:
                        print(f"{model_name.upper()}: Error - {result['error']}")
                
                print("-" * 50)
                
        except KeyboardInterrupt:
            print("\nDemo stopped by user")
        except Exception as e:
            print(f"Demo error: {e}")
        
        print("Real-time prediction demo completed!")


def main():
    """Main function for testing real-time prediction."""
    print("=== Real-Time Music Genre Prediction System ===\n")
    
    # Initialize predictor
    try:
        predictor = RealTimePredictor()
        print("[OK] Real-time predictor initialized successfully\n")
    except Exception as e:
        print(f"Error initializing predictor: {e}")
        return
    
    # Test with a sample audio file if available
    sample_files = [
        "Data/genres_original/blues/blues.00000.wav",
        "Data/genres_original/classical/classical.00000.wav",
        "Data/genres_original/rock/rock.00000.wav"
    ]
    
    print("Testing with sample audio files...")
    for file_path in sample_files:
        if os.path.exists(file_path):
            print(f"\nTesting with: {file_path}")
            result = predictor.predict_from_file(file_path)
            print(f"Prediction: {result['genre']} (confidence: {result['confidence']:.3f})")
            print(f"Latency: {result['latency']:.3f}s")
            break
    
    # Interactive menu
    while True:
        print("\n=== Real-Time Prediction Menu ===")
        print("1. Record and predict (3 seconds)")
        print("2. Record and predict (10 seconds)")
        print("3. Real-time demo (continuous)")
        print("4. Test with audio file")
        print("5. Compare all models")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == '1':
            try:
                audio = predictor.record_audio(3.0)
                result = predictor.predict_genre(audio)
                print(f"\nPrediction: {result['genre']} (confidence: {result['confidence']:.3f})")
                print(f"Latency: {result['latency']:.3f}s")
            except Exception as e:
                print(f"Error: {e}")
        
        elif choice == '2':
            try:
                audio = predictor.record_audio(10.0)
                result = predictor.predict_genre(audio)
                print(f"\nPrediction: {result['genre']} (confidence: {result['confidence']:.3f})")
                print(f"Latency: {result['latency']:.3f}s")
            except Exception as e:
                print(f"Error: {e}")
        
        elif choice == '3':
            try:
                duration = float(input("Enter demo duration in seconds (default 30): ") or "30")
                predictor.real_time_prediction_demo(duration)
            except Exception as e:
                print(f"Error: {e}")
        
        elif choice == '4':
            file_path = input("Enter path to audio file: ").strip()
            if os.path.exists(file_path):
                result = predictor.predict_from_file(file_path)
                print(f"\nPrediction: {result['genre']} (confidence: {result['confidence']:.3f})")
                print(f"Latency: {result['latency']:.3f}s")
            else:
                print("File not found!")
        
        elif choice == '5':
            try:
                audio = predictor.record_audio(3.0)
                predictions = predictor.compare_models(audio)
                print("\n=== Model Comparison ===")
                for model_name, result in predictions.items():
                    if 'error' not in result:
                        print(f"{model_name.upper()}: {result['genre']} "
                              f"(confidence: {result['confidence']:.3f}, "
                              f"latency: {result['latency']:.3f}s)")
                    else:
                        print(f"{model_name.upper()}: Error - {result['error']}")
            except Exception as e:
                print(f"Error: {e}")
        
        elif choice == '6':
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()