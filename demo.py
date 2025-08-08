#!/usr/bin/env python3
"""
apollodB Demo Script
Test the inference functionality without Streamlit
Built by Parikshit Kumar in California
"""

import numpy as np
import json
from backend.inference import MusicEmotionPredictor

def demo_inference():
    """Demonstrate the inference functionality"""
    print("🎵 apollodB Inference Demo")
    print("=" * 40)
    
    try:
        # Initialize predictor
        print("📚 Loading model...")
        predictor = MusicEmotionPredictor()
        print("✅ Model loaded successfully!")
        
        # Test EQ generation for different emotions
        emotions = ["happy", "sad", "angry", "calm", "neutral"]
        
        print("\n🎛️ Testing EQ Generation:")
        print("-" * 30)
        
        for emotion in emotions:
            print(f"\n{emotion.upper()} Emotion:")
            eq_data = predictor.generate_eq_curves(emotion, aggression=0.7)
            
            print(f"Description: {eq_data['description']}")
            print("Parametric EQ:")
            print(eq_data['parametric'])
            print("Wavelet EQ (first 100 chars):")
            print(eq_data['wavelet'][:100] + "...")
        
        # Test valence-arousal conversion
        print("\n📊 Valence-Arousal Mapping:")
        print("-" * 30)
        
        for emotion in emotions:
            valence, arousal = predictor.emotion_to_va(emotion)
            print(f"{emotion.capitalize()}: Valence={valence:.2f}, Arousal={arousal:.2f}")
        
        print("\n✅ All tests passed successfully!")
        print("\nTo run the full apollodB app:")
        print("   python launch.py")
        print("   or")
        print("   streamlit run app.py")
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        print("\nPlease ensure all model files are present:")
        print("   - best_model.h5")
        print("   - scaler_mean.npy")
        print("   - scaler_scale.npy") 
        print("   - labels.json")

if __name__ == "__main__":
    demo_inference()
