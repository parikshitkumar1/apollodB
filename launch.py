#!/usr/bin/env python3
"""
apollodB Launcher Script
Built by Parikshit Kumar in California
"""

import sys
import subprocess
import os

def check_requirements():
    """Check if required files and dependencies exist"""
    required_files = [
        "best_model.h5",
        "scaler_mean.npy", 
        "scaler_scale.npy",
        "labels.json"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required model files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nPlease ensure all model files are in the current directory.")
        return False
    
    print("✅ All model files found!")
    return True

def install_requirements():
    """Install required packages"""
    try:
        print("📦 Installing requirements...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install requirements. Please install manually:")
        print("   pip install -r requirements.txt")
        return False

def launch_app():
    """Launch the Streamlit app"""
    try:
        print("🚀 Launching apollodB...")
        print("   Open your browser to: http://localhost:8501")
        print("   Press Ctrl+C to stop the application")
        print("-" * 50)
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n👋 apollodB stopped. Thank you for using our app!")
    except Exception as e:
        print(f"❌ Error launching app: {e}")

def main():
    print("🎵 apollodB - Music Emotion Analysis & EQ Optimization")
    print("   Built by Parikshit Kumar in California")
    print("=" * 60)
    
    # Check if model files exist
    if not check_requirements():
        sys.exit(1)
    
    # Install requirements if needed
    try:
        import streamlit
        import tensorflow
        import librosa
        print("✅ Dependencies already installed!")
    except ImportError:
        if not install_requirements():
            sys.exit(1)
    
    # Launch the app
    launch_app()

if __name__ == "__main__":
    main()
