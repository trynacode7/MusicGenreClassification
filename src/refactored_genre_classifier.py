"""
Refactored Music Genre Classification System

This module provides:
1. Batch processing for entire songs
2. Near-real-time streaming simulation with chunked processing
3. Real-time dashboard/UI with genre probability visualization
4. Modular code structure for feature extraction, prediction, and UI

No live microphone input - works only with pre-recorded audio files.
"""

import numpy as np
import pandas as pd
import librosa
import time
import os
import threading
from typing import Dict, List, Tuple, Optional
import joblib
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import seaborn as sns
from scipy.ndimage import zoom

warnings.filterwarnings('ignore')

# Import TensorFlow for CNN if available
try:
    from tensorflow.keras.models import load_model
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not available. CNN predictions will be disabled.")

class MusicGenreClassifier:
    """
    Refactored music genre classification system with batch and streaming capabilities.
    
    Features:
    - Batch processing for entire songs
    - Streaming simulation with chunked processing
    - Real-time dashboard with probability visualization
    - Modular architecture
    """
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize the music genre classifier.
        
        Args:
            models_dir: Directory containing trained models
        """
        self.models_dir = Path(models_dir)
        self.models = {}
        self.scaler = None
        self.label_encoder = None
        self.sample_rate = 22050
        self.chunk_duration = 3.0  # seconds
        self.chunk_overlap = 0.5  # seconds overlap between chunks
        
        # Load models and preprocessing artifacts
        self._load_models()
        
        # Dashboard components
        self.dashboard = None
        self.is_streaming = False
        self.current_file = None
        self.streaming_results = []
        
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
                    self.models['cnn'] = load_model(self.models_dir / "cnn_model.keras")
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
            print(f"Error extracting features: {e}")
            # Return zero features if extraction fails
            return np.zeros(57, dtype=np.float64)  # Expected feature count

    def predict_genre(self, audio_data: np.ndarray, model_name: str = 'svm') -> Dict:
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
            mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=self.sample_rate)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Resize to match training size (128x128)
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

    def batch_process_songs(self, input_dir: str, output_file: str = "batch_results.csv", 
                           model_name: str = 'svm') -> pd.DataFrame:
        """
        Process entire songs in batch mode.
        
        Args:
            input_dir: Directory containing audio files
            output_file: Output CSV file path
            model_name: Model to use for prediction
            
        Returns:
            DataFrame with results
        """
        print(f"Starting batch processing with {model_name} model...")
        print(f"Input directory: {input_dir}")
        print(f"Output file: {output_file}")
        
        results = []
        input_path = Path(input_dir)
        
        # Find all audio files
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a']
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(input_path.rglob(f"*{ext}"))
        
        print(f"Found {len(audio_files)} audio files")
        
        for i, file_path in enumerate(audio_files):
            print(f"Processing {i+1}/{len(audio_files)}: {file_path.name}")
            
            try:
                # Load audio file
                audio_data, sr = librosa.load(file_path, sr=self.sample_rate)
                
                # Extract true genre from directory structure
                true_genre = file_path.parent.name
                
                # Predict genre
                result = self.predict_genre(audio_data, model_name)
                
                # Store results
                results.append({
                    'file_name': file_path.name,
                    'file_path': str(file_path),
                    'true_genre': true_genre,
                    'predicted_genre': result['genre'],
                    'confidence': result['confidence'],
                    'latency': result['latency'],
                    'model': model_name
                })
                
                print(f"  Predicted: {result['genre']} (confidence: {result['confidence']:.3f})")
                
            except Exception as e:
                print(f"  Error processing {file_path.name}: {e}")
                results.append({
                    'file_name': file_path.name,
                    'file_path': str(file_path),
                    'true_genre': 'unknown',
                    'predicted_genre': 'error',
                    'confidence': 0.0,
                    'latency': 0.0,
                    'model': model_name,
                    'error': str(e)
                })
        
        # Create DataFrame and save
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        print(f"Batch processing completed. Results saved to {output_file}")
        
        return df

    def create_audio_chunks(self, audio_data: np.ndarray, sr: int = 22050) -> List[Tuple[np.ndarray, float, float]]:
        """
        Split audio into overlapping chunks for streaming simulation.
        
        Args:
            audio_data: Audio signal
            sr: Sample rate
            
        Returns:
            List of (chunk_audio, start_time, end_time) tuples
        """
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

    def streaming_simulation(self, file_path: str, output_file: str = "streaming_results.csv", 
                           model_name: str = 'svm', chunk_delay: float = 0.1) -> pd.DataFrame:
        """
        Simulate near-real-time processing by processing audio in chunks.
        
        Args:
            file_path: Path to audio file
            output_file: Output CSV file path
            model_name: Model to use for prediction
            chunk_delay: Delay between chunk processing (simulates real-time)
            
        Returns:
            DataFrame with streaming results
        """
        print(f"Starting streaming simulation for {file_path}")
        print(f"Chunk duration: {self.chunk_duration}s, Overlap: {self.chunk_overlap}s")
        
        try:
            # Load audio file
            audio_data, sr = librosa.load(file_path, sr=self.sample_rate)
            
            # Extract true genre from directory structure
            true_genre = Path(file_path).parent.name
            
            # Create chunks
            chunks = self.create_audio_chunks(audio_data, sr)
            print(f"Created {len(chunks)} chunks")
            
            results = []
            
            for i, (chunk_audio, start_time, end_time) in enumerate(chunks):
                print(f"Processing chunk {i+1}/{len(chunks)}: {start_time:.1f}s - {end_time:.1f}s")
                
                # Predict genre for this chunk
                result = self.predict_genre(chunk_audio, model_name)
                
                # Store results
                results.append({
                    'file_name': Path(file_path).name,
                    'chunk_start': start_time,
                    'chunk_end': end_time,
                    'chunk_index': i,
                    'true_genre': true_genre,
                    'predicted_genre': result['genre'],
                    'confidence': result['confidence'],
                    'latency': result['latency'],
                    'model': model_name
                })
                
                print(f"  Chunk prediction: {result['genre']} (confidence: {result['confidence']:.3f})")
                
                # Simulate real-time delay
                time.sleep(chunk_delay)
            
            # Create DataFrame and save
            df = pd.DataFrame(results)
            df.to_csv(output_file, index=False)
            print(f"Streaming simulation completed. Results saved to {output_file}")
            
            return df
            
        except Exception as e:
            print(f"Error in streaming simulation: {e}")
            return pd.DataFrame()

    def create_dashboard(self):
        """Create real-time dashboard for genre probability visualization."""
        self.dashboard = GenreDashboard(self)
        return self.dashboard

    def start_streaming_demo(self, file_path: str, model_name: str = 'svm'):
        """Start streaming demo with dashboard."""
        if self.dashboard is None:
            self.dashboard = self.create_dashboard()
        
        self.dashboard.start_streaming_demo(file_path, model_name)


class GenreDashboard:
    """
    Real-time dashboard for music genre classification visualization.
    
    Features:
    - Genre probability bar chart
    - Streaming simulation with live updates
    - File selection and model comparison
    """
    
    def __init__(self, classifier: MusicGenreClassifier):
        """
        Initialize the dashboard.
        
        Args:
            classifier: MusicGenreClassifier instance
        """
        self.classifier = classifier
        self.root = tk.Tk()
        self.root.title("Music Genre Classification Dashboard")
        self.root.geometry("1200x800")
        
        # Dashboard state
        self.is_streaming = False
        self.current_file = None
        self.streaming_results = []
        self.selected_model = tk.StringVar(value="svm")
        
        # Create UI
        self._create_ui()
        
    def _create_ui(self):
        """Create the dashboard UI."""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # File selection
        ttk.Label(control_frame, text="Audio File:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.file_var = tk.StringVar()
        file_entry = ttk.Entry(control_frame, textvariable=self.file_var, width=50)
        file_entry.grid(row=0, column=1, padx=(0, 5))
        ttk.Button(control_frame, text="Browse", command=self._browse_file).grid(row=0, column=2)
        
        # Model selection
        ttk.Label(control_frame, text="Model:").grid(row=0, column=3, sticky=tk.W, padx=(20, 5))
        model_combo = ttk.Combobox(control_frame, textvariable=self.selected_model, 
                                 values=["random_forest", "svm", "cnn"], state="readonly")
        model_combo.grid(row=0, column=4, padx=(0, 5))
        
        # Buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=1, column=0, columnspan=5, pady=(10, 0))
        
        ttk.Button(button_frame, text="Start Streaming Demo", 
                  command=self._start_streaming).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Batch Process", 
                  command=self._batch_process).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Stop", 
                  command=self._stop_streaming).pack(side=tk.LEFT, padx=(0, 5))
        
        # Results frame
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create matplotlib figure
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, results_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize plots
        self._init_plots()
        
    def _init_plots(self):
        """Initialize the plots."""
        # Genre probability bar chart
        self.ax1.set_title("Genre Probabilities")
        self.ax1.set_xlabel("Genre")
        self.ax1.set_ylabel("Probability")
        self.ax1.set_ylim(0, 1)
        
        # Streaming results plot
        self.ax2.set_title("Streaming Results")
        self.ax2.set_xlabel("Time (seconds)")
        self.ax2.set_ylabel("Confidence")
        
        plt.tight_layout()
        
    def _browse_file(self):
        """Browse for audio file."""
        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[("Audio files", "*.wav *.mp3 *.flac *.m4a"), ("All files", "*.*")]
        )
        if file_path:
            self.file_var.set(file_path)
            
    def _start_streaming(self):
        """Start streaming demo."""
        file_path = self.file_var.get()
        if not file_path or not os.path.exists(file_path):
            messagebox.showerror("Error", "Please select a valid audio file")
            return
            
        self.current_file = file_path
        self.is_streaming = True
        self.streaming_results = []
        
        # Start streaming in separate thread
        threading.Thread(target=self._streaming_worker, daemon=True).start()
        
    def _streaming_worker(self):
        """Worker thread for streaming simulation."""
        try:
            # Load audio file
            audio_data, sr = librosa.load(self.current_file, sr=self.classifier.sample_rate)
            
            # Create chunks
            chunks = self.classifier.create_audio_chunks(audio_data, sr)
            
            for i, (chunk_audio, start_time, end_time) in enumerate(chunks):
                if not self.is_streaming:
                    break
                    
                # Predict genre for this chunk
                result = self.classifier.predict_genre(chunk_audio, self.selected_model.get())
                
                # Store results
                self.streaming_results.append({
                    'chunk_index': i,
                    'start_time': start_time,
                    'end_time': end_time,
                    'genre': result['genre'],
                    'confidence': result['confidence'],
                    'latency': result['latency']
                })
                
                # Update dashboard
                self.root.after(0, self._update_dashboard)
                
                # Simulate real-time delay
                time.sleep(0.1)
                
        except Exception as e:
            print(f"Error in streaming worker: {e}")
            self.root.after(0, lambda: messagebox.showerror("Error", f"Streaming error: {e}"))
            
    def _update_dashboard(self):
        """Update the dashboard with new results."""
        if not self.streaming_results:
            return
            
        # Get latest result
        latest_result = self.streaming_results[-1]
        
        # Update genre probability chart
        self.ax1.clear()
        self.ax1.set_title(f"Genre Probabilities - {latest_result['genre']}")
        self.ax1.set_xlabel("Genre")
        self.ax1.set_ylabel("Probability")
        self.ax1.set_ylim(0, 1)
        
        # Create genre probability bars (simplified - showing confidence as probability)
        genres = self.classifier.label_encoder.classes_
        probabilities = np.zeros(len(genres))
        
        # Find the predicted genre and set its probability
        predicted_genre = latest_result['genre']
        if predicted_genre in genres:
            genre_idx = np.where(genres == predicted_genre)[0][0]
            probabilities[genre_idx] = latest_result['confidence']
        
        bars = self.ax1.bar(genres, probabilities, color='skyblue', alpha=0.7)
        self.ax1.tick_params(axis='x', rotation=45)
        
        # Update streaming results plot
        self.ax2.clear()
        self.ax2.set_title("Streaming Results")
        self.ax2.set_xlabel("Time (seconds)")
        self.ax2.set_ylabel("Confidence")
        
        if len(self.streaming_results) > 1:
            times = [r['start_time'] for r in self.streaming_results]
            confidences = [r['confidence'] for r in self.streaming_results]
            genres = [r['genre'] for r in self.streaming_results]
            
            self.ax2.plot(times, confidences, 'b-o', markersize=4)
            self.ax2.set_ylim(0, 1)
            self.ax2.grid(True, alpha=0.3)
            
            # Add genre labels
            for i, (time, conf, genre) in enumerate(zip(times, confidences, genres)):
                self.ax2.annotate(genre, (time, conf), xytext=(5, 5), 
                                textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        self.canvas.draw()
        
    def _batch_process(self):
        """Start batch processing."""
        file_path = self.file_var.get()
        if not file_path or not os.path.exists(file_path):
            messagebox.showerror("Error", "Please select a valid audio file")
            return
            
        # Get directory containing the file
        input_dir = os.path.dirname(file_path)
        
        # Start batch processing in separate thread
        def batch_worker():
            try:
                results = self.classifier.batch_process_songs(
                    input_dir, 
                    "batch_results.csv", 
                    self.selected_model.get()
                )
                self.root.after(0, lambda: messagebox.showinfo("Success", 
                    f"Batch processing completed. Processed {len(results)} files."))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Batch processing error: {e}"))
        
        threading.Thread(target=batch_worker, daemon=True).start()
        
    def _stop_streaming(self):
        """Stop streaming demo."""
        self.is_streaming = False
        
    def start_streaming_demo(self, file_path: str, model_name: str = 'svm'):
        """Start streaming demo with specified file and model."""
        self.file_var.set(file_path)
        self.selected_model.set(model_name)
        self._start_streaming()
        
    def run(self):
        """Run the dashboard."""
        self.root.mainloop()


def main():
    """Main function for the refactored music genre classifier."""
    print("=== Refactored Music Genre Classification System ===\n")
    
    # Initialize classifier
    try:
        classifier = MusicGenreClassifier()
        print("[OK] Classifier initialized successfully\n")
    except Exception as e:
        print(f"Error initializing classifier: {e}")
        return
    
    # Interactive menu
    while True:
        print("=== Music Genre Classification Menu ===")
        print("1. Batch process songs from directory")
        print("2. Streaming simulation for single file")
        print("3. Launch real-time dashboard")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            # Batch processing
            input_dir = input("Enter directory path containing audio files: ").strip()
            if os.path.exists(input_dir):
                model_name = input("Enter model name (random_forest/svm/cnn, default: svm): ").strip() or "svm"
                output_file = input("Enter output CSV file name (default: batch_results.csv): ").strip() or "batch_results.csv"
                
                try:
                    results = classifier.batch_process_songs(input_dir, output_file, model_name)
                    print(f"\nBatch processing completed!")
                    print(f"Results saved to: {output_file}")
                    print(f"Processed {len(results)} files")
                except Exception as e:
                    print(f"Error in batch processing: {e}")
            else:
                print("Directory not found!")
        
        elif choice == '2':
            # Streaming simulation
            file_path = input("Enter path to audio file: ").strip()
            if os.path.exists(file_path):
                model_name = input("Enter model name (random_forest/svm/cnn, default: svm): ").strip() or "svm"
                output_file = input("Enter output CSV file name (default: streaming_results.csv): ").strip() or "streaming_results.csv"
                
                try:
                    results = classifier.streaming_simulation(file_path, output_file, model_name)
                    print(f"\nStreaming simulation completed!")
                    print(f"Results saved to: {output_file}")
                    print(f"Processed {len(results)} chunks")
                except Exception as e:
                    print(f"Error in streaming simulation: {e}")
            else:
                print("File not found!")
        
        elif choice == '3':
            # Launch dashboard
            try:
                dashboard = classifier.create_dashboard()
                print("Launching real-time dashboard...")
                dashboard.run()
            except Exception as e:
                print(f"Error launching dashboard: {e}")
        
        elif choice == '4':
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
