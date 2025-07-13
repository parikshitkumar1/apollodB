# apollodB - Music Emotion Recognition Inference Script
# Built by Parikshit Kumar in California

import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
import librosa
import librosa.display
import json
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Audio parameters (must match training script)
SR = 22050
DURATION = 30
N_MELS = 128
HOP_LENGTH = 512
MAX_LEN = 1300

# Emotion mapping (Model was trained on 4 emotions - no "angry" class)
EMOTIONS = ["neutral", "happy", "sad", "calm"]

class MusicEmotionPredictor:
    def __init__(self, model_path: str = "best_model.h5", 
                 scaler_mean_path: str = "scaler_mean.npy",
                 scaler_scale_path: str = "scaler_scale.npy",
                 labels_path: str = "labels.json"):
        """Initialize the emotion predictor"""
        self.model = tf.keras.models.load_model(model_path)
        self.scaler_mean = np.load(scaler_mean_path)
        self.scaler_scale = np.load(scaler_scale_path)
        
        # Load emotion labels
        with open(labels_path, 'r') as f:
            self.emotions = json.load(f)
    
    def extract_features(self, audio_path: str) -> np.ndarray:
        """Extract log-mel spectrogram features from audio file"""
        try:
            y, sr = librosa.load(audio_path, sr=SR, mono=True)
            
            # Pad or truncate to fixed duration
            if len(y) > SR * DURATION:
                y = y[:SR * DURATION]
            else:
                y = np.pad(y, (0, SR * DURATION - len(y)), mode='constant')
            
            # Extract mel spectrogram
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
            S_db = librosa.power_to_db(S, ref=np.max).T  # (time, mels)
            
            # Pad or truncate to fixed length
            if S_db.shape[0] < MAX_LEN:
                pad = MAX_LEN - S_db.shape[0]
                S_db = np.pad(S_db, ((0, pad), (0, 0)), mode='constant')
            else:
                S_db = S_db[:MAX_LEN]
            
            return S_db.astype(np.float32)
        
        except Exception as e:
            print(f"Error extracting features from {audio_path}: {e}")
            return np.zeros((MAX_LEN, N_MELS), dtype=np.float32)
    
    def predict_emotion(self, audio_path: str) -> Dict:
        """Predict emotion from audio file with sophisticated neutral bias handling"""
        # Extract features
        features = self.extract_features(audio_path)
        
        # Normalize using training scaler
        features_normalized = (features - self.scaler_mean) / self.scaler_scale
        
        # Add batch dimension
        features_batch = np.expand_dims(features_normalized, axis=0)
        
        # Predict
        predictions = self.model.predict(features_batch, verbose=0)[0]
        
        # Get emotion probabilities
        emotion_probs = {emotion: float(prob) for emotion, prob in zip(self.emotions, predictions)}
        
        # Advanced neutral bias handling (like in reference code)
        predicted_class = np.argmax(predictions)
        predicted_emotion = self.emotions[predicted_class]
        
        # If neutral is predicted but confidence is < 0.93, use non-neutral emotions
        if predicted_emotion == "neutral" and predictions[predicted_class] < 0.93:
            # Get non-neutral emotions and their probabilities
            non_neutral_probs = []
            for i, emotion in enumerate(self.emotions):
                if emotion != "neutral":
                    non_neutral_probs.append((emotion, predictions[i]))
            
            # Sort by probability and get top 2
            non_neutral_probs.sort(key=lambda x: x[1], reverse=True)
            top_emotions = non_neutral_probs[:2]
            
            # Use the highest non-neutral emotion as prediction
            predicted_emotion = top_emotions[0][0]
            secondary_emotion = top_emotions[1][0] if len(top_emotions) > 1 else top_emotions[0][0]
        else:
            # Use normal prediction logic
            sorted_emotions = sorted(emotion_probs.items(), key=lambda x: x[1], reverse=True)
            predicted_emotion = sorted_emotions[0][0]
            secondary_emotion = sorted_emotions[1][0] if len(sorted_emotions) > 1 else sorted_emotions[0][0]
        
        # Convert to valence-arousal space for visualization
        valence, arousal = self.emotion_to_va(predicted_emotion)
        
        return {
            "primary_emotion": predicted_emotion,
            "secondary_emotion": secondary_emotion,
            "emotion_probabilities": emotion_probs,
            "valence": valence,
            "arousal": arousal,
            "confidence": max(emotion_probs.values())
        }
    
    def emotion_to_va(self, emotion: str) -> Tuple[float, float]:
        """Convert emotion to valence-arousal coordinates (0-1 scale)"""
        va_mapping = {
            "neutral": (0.5, 0.5),
            "happy": (0.8, 0.8),
            "sad": (0.2, 0.2),
            "calm": (0.8, 0.2)
        }
        return va_mapping.get(emotion, (0.5, 0.5))
    
    def generate_eq_curves(self, emotion: str, aggression: float = 0.5) -> Dict[str, str]:
        """Generate EQ curves based on emotion and aggression level"""
        
        # Define base frequency points (ISO 1/3 octave bands)
        frequencies = [
            20, 21, 22, 23, 24, 26, 27, 29, 30, 32, 34, 36, 38, 40, 43, 45, 48, 50, 53, 56, 59, 63, 66, 70, 74, 78, 83, 87, 92, 97, 103, 109, 115, 121, 128, 136, 143, 151, 160, 169, 178, 188, 199, 210, 222, 235, 248, 262, 277, 292, 309, 326, 345, 364, 385, 406, 429, 453, 479, 506, 534, 565, 596, 630, 665, 703, 743, 784, 829, 875, 924, 977, 1032, 1090, 1151, 1216, 1284, 1357, 1433, 1514, 1599, 1689, 1784, 1885, 1991, 2103, 2221, 2347, 2479, 2618, 2766, 2921, 3086, 3260, 3443, 3637, 3842, 4058, 4287, 4528, 4783, 5052, 5337, 5637, 5955, 6290, 6644, 7018, 7414, 7831, 8272, 8738, 9230, 9749, 10298, 10878, 11490, 12137, 12821, 13543, 14305, 15110, 15961, 16860, 17809, 18812, 19871
        ]
        
        # Emotion-based EQ profiles
        eq_profiles = {
            "happy": {
                "bass": 1.5, "low_mid": 0.5, "mid": 2.0, "high_mid": 3.0, "treble": 2.5,
                "description": "Bright and energetic with enhanced presence"
            },
            "sad": {
                "bass": -1.0, "low_mid": -2.0, "mid": -1.5, "high_mid": -2.5, "treble": -3.0,
                "description": "Warm and subdued with reduced harshness"
            },
            "calm": {
                "bass": 0.5, "low_mid": -0.5, "mid": -1.0, "high_mid": -1.5, "treble": -2.0,
                "description": "Smooth and relaxed frequency response"
            },
            "neutral": {
                "bass": 0.0, "low_mid": 0.0, "mid": 0.0, "high_mid": 0.0, "treble": 0.0,
                "description": "Flat reference response"
            }
        }
        
        profile = eq_profiles.get(emotion, eq_profiles["neutral"])
        
        # Generate EQ curve
        eq_gains = []
        for freq in frequencies:
            if freq <= 100:  # Bass
                gain = profile["bass"] * aggression
            elif freq <= 500:  # Low-mid
                gain = profile["low_mid"] * aggression
            elif freq <= 2000:  # Mid
                gain = profile["mid"] * aggression
            elif freq <= 8000:  # High-mid
                gain = profile["high_mid"] * aggression
            else:  # Treble
                gain = profile["treble"] * aggression
            
            eq_gains.append(gain)
        
        # Generate Wavelet EQ format
        wavelet_eq = "GraphicEQ: " + "; ".join([f"{freq} {gain:.1f}" for freq, gain in zip(frequencies, eq_gains)])
        
        # Generate parametric EQ (simplified)
        parametric_eq = f"""
        Low Shelf (100Hz): {profile['bass'] * aggression:.1f}dB
        Low Mid (500Hz): {profile['low_mid'] * aggression:.1f}dB
        Mid (2kHz): {profile['mid'] * aggression:.1f}dB
        High Mid (8kHz): {profile['high_mid'] * aggression:.1f}dB
        High Shelf (16kHz): {profile['treble'] * aggression:.1f}dB
        """
        
        return {
            "wavelet": wavelet_eq,
            "parametric": parametric_eq.strip(),
            "description": profile["description"]
        }
    
    def generate_spectrogram(self, audio_path: str, save_path: str = None) -> str:
        """Generate and save spectrogram visualization"""
        try:
            y, sr = librosa.load(audio_path, sr=SR, mono=True)
            
            # Limit to 30 seconds for visualization
            if len(y) > SR * DURATION:
                y = y[:SR * DURATION]
            
            # Create mel spectrogram
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
            S_db = librosa.power_to_db(S, ref=np.max)
            
            # Create the plot
            plt.figure(figsize=(12, 6))
            plt.style.use('dark_background')
            
            librosa.display.specshow(S_db, sr=sr, hop_length=HOP_LENGTH, 
                                   x_axis='time', y_axis='mel', 
                                   cmap='plasma', fmax=8000)
            
            plt.colorbar(format='%+2.0f dB', label='Power (dB)')
            plt.title('Mel Spectrogram', color='white', fontsize=14, fontweight='bold')
            plt.xlabel('Time (s)', color='white')
            plt.ylabel('Frequency (Hz)', color='white')
            plt.tight_layout()
            
            # Set background and colors
            plt.gca().set_facecolor('black')
            plt.gcf().patch.set_facecolor('black')
            
            if save_path:
                plt.savefig(save_path, facecolor='black', edgecolor='none', dpi=150, bbox_inches='tight')
                plt.close()
                return save_path
            else:
                # Save to temporary file
                import tempfile
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                plt.savefig(temp_file.name, facecolor='black', edgecolor='none', dpi=150, bbox_inches='tight')
                plt.close()
                return temp_file.name
                
        except Exception as e:
            print(f"Error generating spectrogram: {e}")
            return None
    
    def apply_eq_to_audio(self, audio_path: str, emotion: str, aggression: float = 0.5) -> bytes:
        """Apply EQ to audio and return processed audio bytes"""
        try:
            import scipy.signal
            
            # Load audio
            y, sr = librosa.load(audio_path, sr=SR, mono=True)
            
            # Get EQ profile
            eq_data = self.generate_eq_curves(emotion, aggression)
            
            # Parse EQ gains (simplified - using parametric approach)
            eq_profiles = {
                "happy": {"bass": 1.5, "low_mid": 0.5, "mid": 2.0, "high_mid": 3.0, "treble": 2.5},
                "sad": {"bass": -1.0, "low_mid": -2.0, "mid": -1.5, "high_mid": -2.5, "treble": -3.0},
                "calm": {"bass": 0.5, "low_mid": -0.5, "mid": -1.0, "high_mid": -1.5, "treble": -2.0},
                "neutral": {"bass": 0.0, "low_mid": 0.0, "mid": 0.0, "high_mid": 0.0, "treble": 0.0}
            }
            
            profile = eq_profiles.get(emotion, eq_profiles["neutral"])
            
            # Apply basic EQ using biquad filters (simplified implementation)
            processed = y.copy()
            
            # Bass shelf (100Hz)
            if profile["bass"] != 0:
                sos_bass = scipy.signal.butter(2, 100, 'highpass', fs=sr, output='sos')
                bass_gain = 10**(profile["bass"] * aggression / 20)
                processed = processed * bass_gain
            
            # High shelf (8kHz) 
            if profile["treble"] != 0:
                sos_treble = scipy.signal.butter(2, 8000, 'lowpass', fs=sr, output='sos')
                treble_gain = 10**(profile["treble"] * aggression / 20)
                high_freq = y - scipy.signal.sosfilt(sos_treble, y)
                processed = processed + high_freq * (treble_gain - 1)
            
            # Normalize to prevent clipping
            processed = processed / np.max(np.abs(processed)) * 0.95
            
            # Convert to int16 for audio export
            processed_int = (processed * 32767).astype(np.int16)
            
            # Convert to bytes
            import io
            import wave
            buffer = io.BytesIO()
            
            with wave.open(buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sr)
                wav_file.writeframes(processed_int.tobytes())
            
            return buffer.getvalue()
            
        except Exception as e:
            print(f"Error applying EQ to audio: {e}")
            return None

def predict_multiple_files(file_paths: List[str], predictor: MusicEmotionPredictor) -> Dict:
    """Analyze multiple audio files and return aggregate results"""
    results = []
    
    for file_path in file_paths:
        result = predictor.predict_emotion(file_path)
        results.append(result)
    
    if not results:
        return {}
    
    # Aggregate emotions
    emotion_counts = {}
    valence_values = []
    arousal_values = []
    
    for result in results:
        emotion = result["primary_emotion"]
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        valence_values.append(result["valence"])
        arousal_values.append(result["arousal"])
    
    # Determine dominant emotion
    dominant_emotion = max(emotion_counts, key=emotion_counts.get)
    
    # Calculate average valence-arousal
    avg_valence = np.mean(valence_values)
    avg_arousal = np.mean(arousal_values)
    
    return {
        "dominant_emotion": dominant_emotion,
        "emotion_distribution": emotion_counts,
        "average_valence": avg_valence,
        "average_arousal": avg_arousal,
        "individual_results": results,
        "total_songs": len(results)
    }
