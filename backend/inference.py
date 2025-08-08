# apollodB - Music Emotion Recognition Inference Script
# Built by Parikshit Kumar in California

import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
try:
    import tensorflow as tf  # optional for environments without TF
except Exception:
    tf = None
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
        # Load model and scalers if TensorFlow is available; otherwise run in fallback mode
        if tf is not None and os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
        else:
            self.model = None
        self.scaler_mean = None
        self.scaler_scale = None
        if os.path.exists(scaler_mean_path) and os.path.exists(scaler_scale_path):
            try:
                self.scaler_mean = np.load(scaler_mean_path)
                self.scaler_scale = np.load(scaler_scale_path)
            except Exception:
                self.scaler_mean = None
                self.scaler_scale = None
        
        # Load emotion labels
        try:
            with open(labels_path, 'r') as f:
                self.emotions = json.load(f)
        except Exception:
            self.emotions = EMOTIONS

        # Fixed psychoacoustic bands and default Q for musical results
        self.eq_bands = [
            {"name": "sub", "type": "lowshelf", "fc": 45.0, "q": 0.7},
            {"name": "bass", "type": "peaking", "fc": 90.0, "q": 0.9},
            {"name": "low_mids", "type": "peaking", "fc": 250.0, "q": 1.0},
            {"name": "mids", "type": "peaking", "fc": 800.0, "q": 1.0},
            {"name": "presence", "type": "peaking", "fc": 3000.0, "q": 1.1},
            {"name": "brilliance", "type": "highshelf", "fc": 12000.0, "q": 0.7},
        ]

        # Small VA lookup table (valence/arousal on 0 to 1)
        # Gains are in dB and represent gentle shaping. These will be further scaled by aggression and confidence.
        self.va_lut = [
            {"v": 0.2, "a": 0.2, "g": {"sub": 0.0, "bass": -0.5, "low_mids": +0.5, "mids": +0.5, "presence": -0.5, "brilliance": -0.5}},
            {"v": 0.8, "a": 0.2, "g": {"sub": +1.0, "bass": +1.5, "low_mids": -0.5, "mids": -0.5, "presence": +0.5, "brilliance": +0.5}},
            {"v": 0.2, "a": 0.8, "g": {"sub": -0.5, "bass": 0.0, "low_mids": -0.5, "mids": +0.5, "presence": +1.5, "brilliance": +1.0}},
            {"v": 0.8, "a": 0.8, "g": {"sub": +0.5, "bass": +1.0, "low_mids": -0.5, "mids": 0.0, "presence": +1.0, "brilliance": +1.0}},
            {"v": 0.5, "a": 0.5, "g": {"sub": 0.0, "bass": 0.0, "low_mids": 0.0, "mids": 0.0, "presence": 0.0, "brilliance": 0.0}},
        ]

        # Safety limits
        self.eq_limits = {"boost_db_max": 6.0, "cut_db_max": 8.0, "q_min": 0.5, "q_max": 1.6, "hi_total_boost_cap_db": 3.0}
    
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
        # Fallback if model or scalers are unavailable: simple heuristic, default to neutral
        if self.model is None or self.scaler_mean is None or self.scaler_scale is None:
            try:
                y, sr = librosa.load(audio_path, sr=SR, mono=True)
                # quick spectral centroid to pick between calm/happy/sad
                centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
                norm_c = float(np.clip(centroid / (sr/2), 0.0, 1.0))
                if norm_c > 0.6:
                    primary = "happy"
                    secondary = "calm"
                elif norm_c < 0.3:
                    primary = "sad"
                    secondary = "calm"
                else:
                    primary = "neutral"
                    secondary = "calm"
            except Exception:
                primary, secondary = "neutral", "calm"
            valence, arousal = self.emotion_to_va(primary)
            emotion_probs = {e: (0.55 if e == primary else 0.15) for e in self.emotions}
            return {
                "primary_emotion": primary,
                "secondary_emotion": secondary,
                "emotion_probabilities": emotion_probs,
                "valence": valence,
                "arousal": arousal,
                "confidence": max(emotion_probs.values()),
            }

        # Normal path: model available
        features = self.extract_features(audio_path)
        # Normalize using training scaler
        features_normalized = (features - self.scaler_mean) / self.scaler_scale
        # Add batch dimension
        features_batch = np.expand_dims(features_normalized, axis=0)
        # Predict
        predictions = self.model.predict(features_batch, verbose=0)[0]
        emotion_probs = {emotion: float(prob) for emotion, prob in zip(self.emotions, predictions)}
        # Advanced neutral bias handling
        predicted_class = np.argmax(predictions)
        predicted_emotion = self.emotions[predicted_class]
        if predicted_emotion == "neutral" and predictions[predicted_class] < 0.93:
            non_neutral_probs = []
            for i, emotion in enumerate(self.emotions):
                if emotion != "neutral":
                    non_neutral_probs.append((emotion, predictions[i]))
            non_neutral_probs.sort(key=lambda x: x[1], reverse=True)
            top_emotions = non_neutral_probs[:2]
            predicted_emotion = top_emotions[0][0]
            secondary_emotion = top_emotions[1][0] if len(top_emotions) > 1 else top_emotions[0][0]
        else:
            sorted_emotions = sorted(emotion_probs.items(), key=lambda x: x[1], reverse=True)
            predicted_emotion = sorted_emotions[0][0]
            secondary_emotion = sorted_emotions[1][0] if len(sorted_emotions) > 1 else sorted_emotions[0][0]
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
        """Generate EQ curves using VA interpolation and safe constraints. No retraining needed."""

        # Map emotion to VA
        v, a = self.emotion_to_va(emotion)

        # Interpolate band gains from LUT using inverse distance weighting
        def interp_va(vv: float, aa: float) -> dict:
            import math
            weights = []
            for node in self.va_lut:
                dv = vv - node["v"]
                da = aa - node["a"]
                d = math.hypot(dv, da)
                w = 1e9 if d == 0 else 1.0 / (d + 1e-6)
                weights.append((w, node["g"]))
            # weighted sum
            bands = {b["name"]: 0.0 for b in self.eq_bands}
            W = sum(w for w, _ in weights)
            for w, g in weights:
                for k in bands.keys():
                    bands[k] += (g.get(k, 0.0) * w) / (W + 1e-9)
            return bands

        base_gains = interp_va(v, a)

        # Confidence unknown at this stage in this function, assume mid confidence
        conf = 0.7
        scale = max(0.3, min(1.0, conf)) * float(aggression)
        gains = {k: base_gains[k] * scale for k in base_gains}

        # Clamp and smooth between adjacent bands (simple 1D smoothing)
        order = [b["name"] for b in self.eq_bands]
        smoothed = gains.copy()
        for i in range(1, len(order)-1):
            k = order[i]
            km1 = order[i-1]
            kp1 = order[i+1]
            smoothed[k] = (gains[k] * 0.6) + (gains[km1] * 0.2) + (gains[kp1] * 0.2)
        gains = smoothed

        # Enforce limits and cap total high band boost
        for k, g in gains.items():
            if g >= 0:
                gains[k] = min(g, self.eq_limits["boost_db_max"])
            else:
                gains[k] = max(g, -self.eq_limits["cut_db_max"])
        hi_sum = max(0.0, gains.get("presence", 0.0)) + max(0.0, gains.get("brilliance", 0.0))
        if hi_sum > self.eq_limits["hi_total_boost_cap_db"]:
            factor = self.eq_limits["hi_total_boost_cap_db"] / (hi_sum + 1e-9)
            gains["presence"] *= factor
            gains["brilliance"] *= factor

        # Build a full frequency response for Wavelet export by linear interp between band centers
        frequencies = [
            20, 21, 22, 23, 24, 26, 27, 29, 30, 32, 34, 36, 38, 40, 43, 45, 48, 50, 53, 56, 59, 63, 66, 70, 74, 78, 83, 87, 92, 97, 103, 109, 115, 121, 128, 136, 143, 151, 160, 169, 178, 188, 199, 210, 222, 235, 248, 262, 277, 292, 309, 326, 345, 364, 385, 406, 429, 453, 479, 506, 534, 565, 596, 630, 665, 703, 743, 784, 829, 875, 924, 977, 1032, 1090, 1151, 1216, 1284, 1357, 1433, 1514, 1599, 1689, 1784, 1885, 1991, 2103, 2221, 2347, 2479, 2618, 2766, 2921, 3086, 3260, 3443, 3637, 3842, 4058, 4287, 4528, 4783, 5052, 5337, 5637, 5955, 6290, 6644, 7018, 7414, 7831, 8272, 8738, 9230, 9749, 10298, 10878, 11490, 12137, 12821, 13543, 14305, 15110, 15961, 16860, 17809, 18812, 19871
        ]
        # Create points from band centers
        pts_x = [b["fc"] for b in self.eq_bands]
        pts_y = [gains[b["name"]] for b in self.eq_bands]
        # Simple log frequency linear interpolation
        import numpy as np
        lx = np.log10(np.array(pts_x))
        ly = np.array(pts_y)
        fx = np.log10(np.array(frequencies))
        eq_env = np.interp(fx, lx, ly, left=ly[0], right=ly[-1])
        wavelet_eq = "GraphicEQ: " + "; ".join([f"{f} {g:.2f}" for f, g in zip(frequencies, eq_env.tolist())])

        # Parametric summary
        lines = []
        for b in self.eq_bands:
            g = gains[b["name"]]
            if abs(g) < 0.25:
                continue
            q = min(max(b["q"], self.eq_limits["q_min"]), self.eq_limits["q_max"])
            lines.append(f"{b['type'].title()} {int(b['fc'])} Hz: {g:+.1f} dB, Q={q:.2f}")
        parametric_eq = "\n".join(lines) if lines else "Flat"

        description = {
            "happy": "Bright with enhanced presence",
            "sad": "Warm with softened highs",
            "calm": "Smooth and relaxed",
            "neutral": "Flat reference"
        }.get(emotion, "Flat reference")

        return {"wavelet": wavelet_eq, "parametric": parametric_eq, "description": description}
    
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
    
    def apply_eq_to_audio(self, audio_path: str, emotion: str, aggression: float = 0.5,
                          confidence: float = None, warmth: float = 0.0, presence: float = 0.0, air: float = 0.0,
                          hq_linear: bool = False):
        """Apply EQ with VA LUT and content-aware tweaks plus safety/fidelity polish.
        Optional nudges: confidence (0-1), and macro sliders warmth/presence/air in +/-1.5 dB range.
        Preserves original SR/channels/bit depth.
        """
        try:
            import io
            import scipy.signal
            import soundfile as sf
            import numpy as np

            # Probe original info to preserve quality
            info = sf.info(audio_path)
            orig_sr = info.samplerate
            orig_channels = info.channels
            orig_subtype = info.subtype or 'PCM_24'

            # Load without resampling, preserve channels (shape: frames x channels)
            y, sr = sf.read(audio_path, always_2d=True)
            if sr != orig_sr:
                orig_sr = sr

            # Compute content features for gentle rule based modifiers
            def content_features(sig: np.ndarray, sr: int) -> dict:
                # Simple spectral bands using FFT
                X = np.fft.rfft(sig.astype(np.float64))
                freqs = np.fft.rfftfreq(sig.shape[0], d=1.0/sr)
                def band_rms(lo, hi):
                    m = (freqs >= lo) & (freqs < hi)
                    if not np.any(m):
                        return 0.0
                    return float(np.sqrt(np.mean(np.abs(X[m])**2)) + 1e-12)
                bass = band_rms(20, 120)
                lowmid = band_rms(120, 400)
                presence = band_rms(1600, 5000)
                air = band_rms(8000, min(16000, sr/2 - 1))
                full = band_rms(20, min(20000, sr/2 - 1))
                centroid = float(np.sum(freqs * np.abs(X)) / (np.sum(np.abs(X)) + 1e-12))
                return {
                    "bass_ratio": bass / (lowmid + 1e-9),
                    "presence_ratio": presence / (lowmid + 1e-9),
                    "air_ratio": air / (presence + 1e-9),
                    "centroid": centroid,
                    "full": full,
                }

            # Map emotion to VA and interpolate band gains
            v, a = self.emotion_to_va(emotion)
            def interp_va(vv: float, aa: float) -> dict:
                import math
                weights = []
                for node in self.va_lut:
                    dv = vv - node["v"]
                    da = aa - node["a"]
                    d = math.hypot(dv, da)
                    w = 1e9 if d == 0 else 1.0 / (d + 1e-6)
                    weights.append((w, node["g"]))
                bands = {b["name"]: 0.0 for b in self.eq_bands}
                W = sum(w for w, _ in weights)
                for w, g in weights:
                    for k in bands.keys():
                        bands[k] += (g.get(k, 0.0) * w) / (W + 1e-9)
                return bands

            base_gains = interp_va(v, a)
            # Confidence-weighted aggression
            conf = 0.7 if (confidence is None) else float(np.clip(confidence, 0.0, 1.0))
            scale = max(0.3, min(1.0, conf)) * float(aggression)

            # Content aware modifiers measured on a mono mixdown for analysis
            mono = np.mean(y, axis=1)
            feats = content_features(mono, orig_sr)
            mods = {k: 0.0 for k in base_gains}
            if feats["bass_ratio"] > 1.4:
                mods["bass"] -= min(2.0, (feats["bass_ratio"] - 1.4) * 2.0)
            if feats["presence_ratio"] > 1.3:
                mods["presence"] -= min(2.0, (feats["presence_ratio"] - 1.3) * 2.0)
            if feats["air_ratio"] > 1.3:
                mods["brilliance"] -= min(1.5, (feats["air_ratio"] - 1.3) * 1.5)

            # Gentle loudness-aware tilt (approximate): derive overall level and apply <= +/-1.5 dB tilt
            # Use simple RMS in dBFS as proxy
            rms = float(np.sqrt(np.mean(mono**2)) + 1e-12)
            dbfs = 20.0 * np.log10(rms + 1e-12)
            # Reference around -16 dBFS; if much lower, add slight bass/treble compensation; if higher, temper highs
            tilt = float(np.clip((-16.0 - dbfs) / 24.0, -0.75, 0.75))  # negative when loud, positive when quiet
            # Map tilt to shelves (clamped to +/-1.5 dB overall)
            mods["sub"] += float(np.clip(tilt * 2.0, -1.5, 1.5))
            mods["brilliance"] += float(np.clip(tilt * 1.5, -1.5, 1.5))

            # Spectral feature refinement (very subtle): use centroid to gate presence/air tweaks
            try:
                centroid = float(librosa.feature.spectral_centroid(y=mono, sr=orig_sr).mean())
                ny = orig_sr / 2.0
                c_norm = float(np.clip(centroid / (ny + 1e-9), 0.0, 1.0))
                if c_norm > 0.55:
                    mods["presence"] -= min(1.5, (c_norm - 0.55) * 3.0)
                elif c_norm < 0.35:
                    mods["brilliance"] += min(1.5, (0.35 - c_norm) * 3.0)
            except Exception:
                pass

            # Optional macro nudges (clamped to +/-1.5 dB) applied after VA+mods, before limits
            macro = {
                "sub": float(np.clip(warmth, -1.5, 1.5)),
                "bass": float(np.clip(warmth * 0.5, -0.75, 0.75)),
                "presence": float(np.clip(presence, -1.5, 1.5)),
                "brilliance": float(np.clip(air, -1.5, 1.5)),
            }

            gains = {k: (base_gains[k] + mods.get(k, 0.0)) * scale for k in base_gains}
            for k in gains:
                gains[k] += macro.get(k, 0.0)
            # Smooth adjacent bands
            order = [b["name"] for b in self.eq_bands]
            smoothed = gains.copy()
            for i in range(1, len(order)-1):
                k = order[i]
                km1 = order[i-1]
                kp1 = order[i+1]
                smoothed[k] = (gains[k] * 0.6) + (gains[km1] * 0.2) + (gains[kp1] * 0.2)
            gains = smoothed

            # Enforce limits and cap high total boost
            for k, g in gains.items():
                if g >= 0:
                    gains[k] = min(g, self.eq_limits["boost_db_max"])
                else:
                    gains[k] = max(g, -self.eq_limits["cut_db_max"])
            hi_sum = max(0.0, gains.get("presence", 0.0)) + max(0.0, gains.get("brilliance", 0.0))
            if hi_sum > self.eq_limits["hi_total_boost_cap_db"]:
                factor = self.eq_limits["hi_total_boost_cap_db"] / (hi_sum + 1e-9)
                gains["presence"] *= factor
                gains["brilliance"] *= factor

            # Biquad filter designers (RBJ cookbook)
            def biquad_peaking(fs, f0, Q, gain_db):
                A = 10 ** (gain_db / 40)
                w0 = 2 * np.pi * f0 / fs
                alpha = np.sin(w0) / (2 * max(Q, 1e-6))
                cosw0 = np.cos(w0)
                b0 = 1 + alpha * A
                b1 = -2 * cosw0
                b2 = 1 - alpha * A
                a0 = 1 + alpha / A
                a1 = -2 * cosw0
                a2 = 1 - alpha / A
                b = np.array([b0, b1, b2]) / a0
                a = np.array([1.0, a1 / a0, a2 / a0])
                return scipy.signal.tf2sos(b, a)

            def biquad_lowshelf(fs, f0, Q, gain_db):
                A = 10 ** (gain_db / 40)
                w0 = 2 * np.pi * f0 / fs
                alpha = np.sin(w0) / (2 * max(Q, 1e-6))
                cosw0 = np.cos(w0)
                beta = np.sqrt(A) / max(Q, 1e-6)
                b0 =    A*((A+1) - (A-1)*cosw0 + 2*np.sqrt(A)*alpha)
                b1 =  2*A*((A-1) - (A+1)*cosw0)
                b2 =    A*((A+1) - (A-1)*cosw0 - 2*np.sqrt(A)*alpha)
                a0 =       (A+1) + (A-1)*cosw0 + 2*np.sqrt(A)*alpha
                a1 =   -2*((A-1) + (A+1)*cosw0)
                a2 =       (A+1) + (A-1)*cosw0 - 2*np.sqrt(A)*alpha
                b = np.array([b0, b1, b2]) / a0
                a = np.array([1.0, a1 / a0, a2 / a0])
                return scipy.signal.tf2sos(b, a)

            def biquad_highshelf(fs, f0, Q, gain_db):
                A = 10 ** (gain_db / 40)
                w0 = 2 * np.pi * f0 / fs
                alpha = np.sin(w0) / (2 * max(Q, 1e-6))
                cosw0 = np.cos(w0)
                b0 =    A*((A+1) + (A-1)*cosw0 + 2*np.sqrt(A)*alpha)
                b1 = -2*A*((A-1) + (A+1)*cosw0)
                b2 =    A*((A+1) + (A-1)*cosw0 - 2*np.sqrt(A)*alpha)
                a0 =       (A+1) - (A-1)*cosw0 + 2*np.sqrt(A)*alpha
                a1 =    2*((A-1) - (A+1)*cosw0)
                a2 =       (A+1) - (A-1)*cosw0 - 2*np.sqrt(A)*alpha
                b = np.array([b0, b1, b2]) / a0
                a = np.array([1.0, a1 / a0, a2 / a0])
                return scipy.signal.tf2sos(b, a)

            # Preamp headroom: compute planned max positive gain and attenuate pre-filter
            planned_max = max(0.0, max([g for g in gains.values() if g is not None] + [0.0]))
            preamp_db = -planned_max if planned_max > 0 else 0.0
            preamp = float(10 ** (preamp_db / 20.0))
            if preamp_db != 0.0:
                y = y * preamp

            # Choose processing path: IIR (fast) or Linear-phase FIR (HQ)
            processed = y.copy()
            if not hq_linear:
                # IIR SOS chain
                sos_chain = []
                for band in self.eq_bands:
                    g = gains.get(band["name"], 0.0)
                    if abs(g) < 0.25:
                        continue
                    q = float(min(max(band["q"], self.eq_limits["q_min"]), self.eq_limits["q_max"]))
                    if band["type"] == "peaking":
                        sos = biquad_peaking(orig_sr, band["fc"], q, g)
                    elif band["type"] == "lowshelf":
                        sos = biquad_lowshelf(orig_sr, band["fc"], q, g)
                    else:  # highshelf
                        sos = biquad_highshelf(orig_sr, band["fc"], q, g)
                    sos_chain.append(sos)
                if sos_chain:
                    sos_all = np.vstack(sos_chain)
                    for ch in range(processed.shape[1]):
                        processed[:, ch] = scipy.signal.sosfilt(sos_all, processed[:, ch])
            else:
                # Linear-phase FIR: build target magnitude response via band interpolation, then firwin2
                import numpy as np
                import scipy.signal as sps
                # Frequency grid 0..Nyquist (normalized 0..1)
                ny = orig_sr / 2.0
                # Use same interpolation approach as generate_eq_curves
                pts_x = np.array([b["fc"] for b in self.eq_bands])
                pts_y_db = np.array([gains[b["name"]] for b in self.eq_bands])
                fx = np.log10(np.clip(pts_x, 20, ny))
                # Dense log grid
                grid = np.geomspace(20.0, max(100.0, ny-1.0), 1024)
                grid_log = np.log10(grid)
                env_db = np.interp(grid_log, fx, pts_y_db, left=pts_y_db[0], right=pts_y_db[-1])
                env_lin = 10 ** (env_db / 20.0)
                # Map to firwin2 inputs (normalized freq 0..1)
                freqs_norm = np.concatenate(([0.0], grid / ny, [1.0]))
                gains_lin = np.concatenate(([env_lin[0]], env_lin, [env_lin[-1]]))
                # FIR length tradeoff (short enough for speed, long enough for shape)
                numtaps = int(min(max(orig_sr * 0.02, 1024), 8192))  # ~20ms, clamp
                numtaps = numtaps + (numtaps % 2 == 0)  # make odd for exact linear phase
                try:
                    fir = sps.firwin2(numtaps, freqs_norm, gains_lin, window='hann')
                except Exception:
                    # Fallback: flat
                    fir = sps.firwin(numtaps, 0.99, window='hann')
                for ch in range(processed.shape[1]):
                    processed[:, ch] = sps.fftconvolve(processed[:, ch], fir, mode='same')

            # Prevent clipping with true-peak safety (simple oversampled peak check)
            peak = float(np.max(np.abs(processed)))
            if peak > 0.999:
                processed = processed / peak * 0.999
            isp = None
            try:
                # 4x oversample for ISP check per channel
                os_factor = 4
                import scipy.signal
                os_peaks = []
                for ch in range(processed.shape[1]):
                    up = scipy.signal.resample_poly(processed[:, ch], os_factor, 1)
                    os_peaks.append(np.max(np.abs(up)))
                isp = float(np.max(os_peaks))
                if isp > 1.0:
                    processed = processed / isp * 0.999
            except Exception:
                pass

            # Choose a valid WAV subtype regardless of source container
            valid_wav_subtypes = {"PCM_16", "PCM_24", "PCM_32", "FLOAT", "DOUBLE"}
            write_subtype = orig_subtype if (orig_subtype in valid_wav_subtypes) else "PCM_24"

            # Optional TPDF dither if writing to integer PCM from float domain
            if write_subtype in {"PCM_16", "PCM_24"}:
                bits = 16 if write_subtype == "PCM_16" else 24
                lsb = 1.0 / (2 ** (bits - 1))
                dither = (np.random.rand(*processed.shape) - 0.5 + np.random.rand(*processed.shape) - 0.5) * lsb
                processed = np.clip(processed + dither, -1.0, 1.0)

            # Optional TPDF dither if writing to integer PCM from float domain
            if write_subtype in {"PCM_16", "PCM_24"}:
                bits = 16 if write_subtype == "PCM_16" else 24
                lsb = 1.0 / (2 ** (bits - 1))
                dither = (np.random.rand(*processed.shape) - 0.5 + np.random.rand(*processed.shape) - 0.5) * lsb
                processed = np.clip(processed + dither, -1.0, 1.0)

            # Loudness (LUFS) estimate if available
            lufs = None
            try:
                import pyloudnorm as pyln
                meter = pyln.Meter(orig_sr)
                # collapse to mono energy-equivalent for measurement
                mono_proc = processed.mean(axis=1)
                lufs = float(meter.integrated_loudness(mono_proc))
            except Exception:
                lufs = None

            # Write to WAV bytes, preserving samplerate/channels with safe subtype
            buf = io.BytesIO()
            with sf.SoundFile(buf, mode='w', samplerate=orig_sr, channels=orig_channels, subtype=write_subtype, format='WAV') as f:
                f.write(processed)
            buf.seek(0)
            stats = {}
            if isp is not None and isp > 0:
                stats["true_peak_dbtp"] = 20.0 * float(np.log10(isp + 1e-12))
            if lufs is not None:
                stats["lufs"] = lufs
            return buf.getvalue(), stats

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
