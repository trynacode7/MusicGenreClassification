"""
Audio Dashboard Launcher for Music Genre Classification (Current Models)

This script launches the Streamlit audio dashboard that accepts .wav files
and processes them using the models available in your models folder.
"""

import os
import sys
import io
import subprocess
from pathlib import Path

# Force UTF-8 encoding for stdout and stderr
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'streamlit', 'plotly', 'pandas', 'numpy', 
        'librosa', 'sklearn', 'matplotlib', 'tensorflow'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"[WARNING] Missing packages: {', '.join(missing_packages)}")
        return False
    
    print("[OK] All required dependencies available")
    return True

def check_models():
    """Check if required model files exist. Warn instead of blocking."""
    models_dir = Path("models")
    required_files = [
        "label_encoder.pkl",
        "random_forest_model.pkl",
        "svm_model.pkl",
        "cnn_model_transfer.keras",
        "cnn_label_encoder.pkl",
        "scaler.pkl"  # use scaler.pkl instead of scaler_improved.pkl
    ]
    
    missing_files = []
    for file in required_files:
        if not (models_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"[WARNING] Missing model files: {', '.join(missing_files)}")
        print("The dashboard will attempt to run with available models.")
        return True  # allow launch even if some models are missing
    
    print("✓ All required model files found")
    return True

def install_dependencies():
    """Install required dependencies."""
    print("Installing dependencies...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "-r", "requirements.txt"
        ], check=True)
        print("✓ Dependencies installed successfully")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to install dependencies: {e}")
        return False

def run_audio_dashboard():
    """Run the audio dashboard."""
    print("Launching Music Genre Classification Audio Dashboard...")
    print("The dashboard will open in your browser at: http://localhost:8501")
    print("\nFeatures:")
    print("- Upload .wav audio files directly")
    print("- Real-time streaming simulation")
    print("- Genre probability visualization")
    print("- Spectrogram and waveform display")
    print("- Multiple model support (Random Forest, SVM, CNN)")
    
    try:
        # Change to the project directory
        os.chdir(Path(__file__).parent)
        
        # Run streamlit dashboard
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "src/streamlit_audio_dashboard.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ], check=True)
    except Exception as e:
        print(f"[ERROR] Could not launch dashboard: {e}")

def main():
    """Main launcher function."""
    print("=== Music Genre Classification Audio Dashboard Launcher ===\n")
    
    # Check dependencies
    if not check_dependencies():
        print("\nInstalling missing dependencies...")
        if not install_dependencies():
            print("Please install dependencies manually:")
            print("pip install -r requirements.txt")
    
    # Check models
    check_models()
    
    print("\nAll checks complete! Launching dashboard...\n")
    
    try:
        run_audio_dashboard()
    except KeyError as e:
        print(f"[WARNING] Missing model key {e}. Continuing with available models...")
        run_audio_dashboard()

if __name__ == "__main__":
    main()
