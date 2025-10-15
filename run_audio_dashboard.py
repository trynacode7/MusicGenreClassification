"""
Audio Dashboard Launcher for Music Genre Classification (Improved Models)

This script launches the Streamlit audio dashboard that accepts .wav files
and processes them using pre-trained improved models.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'streamlit', 'plotly', 'pandas', 'numpy', 
        'librosa', 'scikit-learn', 'matplotlib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}")
        return False
    
    print("✓ All required dependencies available")
    return True

def check_models():
    """Check if improved model files exist."""
    models_dir = Path("models")
    required_files = [
        "scaler_improved.pkl",
        "label_encoder_improved.pkl",
        "svm_improved.pkl",
        "random_forest_improved.pkl"
    ]
    
    missing_files = []
    for file in required_files:
        if not (models_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"Missing improved model files: {', '.join(missing_files)}")
        print("Please run model training first:")
        print("python src/model_training.py")
        return False
    
    print("✓ All required improved model files found")
    return True

def install_dependencies():
    """Install required dependencies."""
    print("Installing dependencies...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "-r", "requirements_audio_dashboard.txt"
        ])
        print("✓ Dependencies installed successfully")
        return True
    except Exception as e:
        print(f"Error installing dependencies: {e}")
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
    print("- Multiple model support (SVM, Random Forest, CNN)")
    
    try:
        # Change to the project directory
        os.chdir(Path(__file__).parent)
        
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "src/streamlit_audio_dashboard.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except Exception as e:
        print(f"Error launching dashboard: {e}")

def main():
    """Main launcher function."""
    print("=== Music Genre Classification Audio Dashboard Launcher ===\n")
    
    # Check dependencies
    if not check_dependencies():
        print("\nInstalling missing dependencies...")
        if not install_dependencies():
            print("Failed to install dependencies. Please install manually:")
            print("pip install -r requirements_audio_dashboard.txt")
            return
    
    # Check models
    if not check_models():
        print("\nPlease train improved models first:")
        print("python src/model_training.py")
        return
    
    print("\nAll checks passed! Launching dashboard...")
    run_audio_dashboard()

if __name__ == "__main__":
    main()
