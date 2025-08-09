import os
import json
import numpy as np
import torch
import torch.nn as nn
import librosa
import librosa.display  # needed for specshow
import soundfile as sf
from typing import Dict, Optional, Tuple
import torch.nn.functional as F
import tempfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal

# Audio processing parameters
SR = 22050
DURATION = 30
N_MELS = 128
HOP_LENGTH = 512
MAX_LEN = 1300

class SelfAttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Parameter(torch.randn(input_dim, 1))
        nn.init.xavier_uniform_(self.W)
    
    def forward(self, x):
        scores = torch.matmul(x, self.W)
        weights = F.softmax(scores, dim=1)
        return torch.sum(x * weights, dim=1)

class TransformerEncoder(nn.Module):
    def __init__(self, dim, num_heads, ff_dim, dropout=0.0):
        super(TransformerEncoder, self).__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim)
        )
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x):
        x_norm = self.norm1(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.dropout1(attn_output)
        x_norm = self.norm2(x)
        ff_output = self.ff(x_norm)
        x = x + self.dropout2(ff_output)
        return x

class MusicEmotionClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MusicEmotionClassifier, self).__init__()
        self.initial_dense = nn.Linear(input_dim, 128)
        self.gelu = nn.GELU()
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoder(dim=128, num_heads=8, ff_dim=256, dropout=0.2)
            for _ in range(4)
        ])
        self.attn_pool = SelfAttentionPooling(128)
        self.dense = nn.Linear(128, 128)
        self.dropout = nn.Dropout(0.3)
        self.output = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.initial_dense(x)
        x = self.gelu(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.attn_pool(x)
        x = self.dense(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.output(x)
        return x

class MusicEmotionPredictor:
    def __init__(self, model_path: str, labels_path: str, scaler_mean_path: str, scaler_scale_path: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load labels
        with open(labels_path, 'r') as f:
            self.labels = json.load(f)
        self.num_classes = len(self.labels)
        
        # Load scaler parameters
        self.scaler_mean = np.load(scaler_mean_path)
        self.scaler_scale = np.load(scaler_scale_path)
        
        # Initialize model
        self.model = MusicEmotionClassifier(input_dim=N_MELS, num_classes=self.num_classes).to(self.device)
        
        # Load weights
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        print(f"Loaded model with {sum(p.numel() for p in self.model.parameters())} parameters")
        print(f"Available emotion classes: {self.labels}")
    
    def _extract_features(self, audio_path: str) -> np.ndarray:
        """Extract log-mel spectrogram features."""
        # Load audio
        y, sr = librosa.load(audio_path, sr=SR, mono=True)
        
        # Trim or pad to target duration
        if len(y) > SR * DURATION:
            y = y[:SR * DURATION]
        else:
            y = np.pad(y, (0, SR * DURATION - len(y)), mode='constant')
        
        # Extract mel spectrogram
        S = librosa.feature.melspectrogram(
            y=y, sr=SR, n_mels=N_MELS, hop_length=HOP_LENGTH,
            fmin=20, fmax=SR//2, n_fft=2048
        )
        
        # Convert to log scale (dB)
        S_db = librosa.power_to_db(S, ref=np.max)
        
        # Pad/cut to fixed length
        if S_db.shape[1] < MAX_LEN:
            pad = MAX_LEN - S_db.shape[1]
            S_db = np.pad(S_db, ((0, 0), (0, pad)), mode='constant')
        else:
            S_db = S_db[:, :MAX_LEN]
        
        # Transpose to (time, mels)
        return S_db.T.astype(np.float32)
    
    def predict_emotion(self, audio_path: str) -> Dict:
        """Predict emotion from audio file."""
        try:
            # Extract features
            features = self._extract_features(audio_path)
            
            # Scale features
            features = (features - self.scaler_mean) / self.scaler_scale
            
            # Convert to tensor and add batch dimension
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            
            # Get model prediction
            with torch.no_grad():
                logits = self.model(features_tensor)
                probs = torch.softmax(logits, dim=1)
                confidence, pred_idx = torch.max(probs, dim=1)
                
            # Convert to Python native types
            raw_probs = probs.squeeze().cpu().numpy().astype(float)
            idx_sorted = np.argsort(-raw_probs)
            top1 = int(idx_sorted[0])
            top2 = int(idx_sorted[1]) if len(idx_sorted) > 1 else top1
            label_top1 = self.labels[top1]
            label_top2 = self.labels[top2]
            p_top1 = float(raw_probs[top1])
            p_top2 = float(raw_probs[top2])

            # Secondary prioritization rule: if top1 is neutral with <0.9, pick second as primary
            primary = label_top1
            secondary = label_top2 if label_top2 != label_top1 else None
            if label_top1.lower() == "neutral" and p_top1 < 0.9 and label_top2 != "neutral":
                primary, secondary = label_top2, label_top1

            confidence = float(max(p_top1 if primary == label_top1 else p_top2, 0.0))

            # Class probability dict
            probabilities = { self.labels[i]: float(p) for i, p in enumerate(raw_probs) }

            # Estimate Valence/Arousal via probability-weighted barycenter over fixed anchors
            anchors = {
                "happy": (0.8, 0.8),
                "sad": (0.2, 0.2),
                "calm": (0.8, 0.2),
                "neutral": (0.5, 0.5),
                "angry": (0.2, 0.8),
            }
            v_sum = 0.0
            a_sum = 0.0
            for i, p in enumerate(raw_probs):
                emo = self.labels[i].lower()
                vx, ax = anchors.get(emo, (0.5, 0.5))
                v_sum += float(p) * vx
                a_sum += float(p) * ax
            valence = float(np.clip(v_sum, 0.0, 1.0))
            arousal = float(np.clip(a_sum, 0.0, 1.0))

            return {
                "emotion": primary,
                "primary_emotion": primary,
                "secondary_emotion": secondary,
                "confidence": confidence,
                "valence": valence,
                "arousal": arousal,
                "probabilities": probabilities,
            }
            
        except Exception as e:
            print(f"Error in predict_emotion: {e}")
            return {
                'error': str(e),
                'primary_emotion': 'neutral',  # Default fallback
                'emotion': 'unknown',
                'confidence': 0.0,
                'probabilities': {l: 0.0 for l in self.labels}
            }
        
    # --- EQ LUT and constants for VA-driven EQ generation ---
    # Emotion anchors in VA space
    
    
    def generate_eq_curves(
        self,
        emotion: str,
        aggression: float = 0.5,
        *,
        valence: Optional[float] = None,
        arousal: Optional[float] = None,
        confidence: Optional[float] = None,
        secondary: Optional[str] = None,
    ) -> Dict[str, str]:
        """Generate EQ curves based on the predicted emotion.
        
        Args:
            emotion: The predicted emotion
            aggression: Strength of the EQ effect (0.0 to 1.0)
            valence: Optional override [0..1]
            arousal: Optional override [0..1]
            
        Returns:
            Dictionary containing EQ curve data in different formats
        """
        # Define anchors (VA) and band model locally to avoid module-level churn
        anchors = {
            "happy": (0.8, 0.8),
            "sad": (0.2, 0.2),
            "calm": (0.8, 0.2),
            "neutral": (0.5, 0.5),
            "angry": (0.2, 0.8),
        }
        eq_bands = [
            {"name": "sub",        "fc": 40.0,    "q": 0.70, "type": "lowshelf"},
            {"name": "bass",       "fc": 120.0,   "q": 0.70, "type": "lowshelf"},
            {"name": "low_mids",   "fc": 400.0,   "q": 1.00, "type": "peaking"},
            {"name": "mids",       "fc": 2000.0,  "q": 1.00, "type": "peaking"},
            {"name": "presence",   "fc": 4000.0,  "q": 1.20, "type": "peaking"},
            {"name": "brilliance", "fc": 12000.0, "q": 0.70, "type": "highshelf"},
        ]
        eq_limits = {
            "boost_db_max": 6.0,
            "cut_db_max": 6.0,
            "hi_total_boost_cap_db": 6.0,
            "q_min": 0.5,
            "q_max": 2.0,
        }
        # VA LUT nodes define characteristic per-band gains for VA corners/emotions
        va_lut = [
            {"name": "neutral", "v": 0.5, "a": 0.5, "g": {"sub": 0.0, "bass": 0.0, "low_mids": 0.0, "mids": 0.0, "presence": 0.0, "brilliance": 0.0}},
            {"name": "happy",   "v": 0.8, "a": 0.8, "g": {"sub": 0.0, "bass": 0.0, "low_mids": 0.0, "mids": +1.0, "presence": +3.0, "brilliance": +2.0}},
            {"name": "sad",     "v": 0.2, "a": 0.2, "g": {"sub": 0.0, "bass": +1.5, "low_mids": 0.0, "mids": -1.0, "presence": 0.0, "brilliance": 0.0}},
            {"name": "calm",    "v": 0.8, "a": 0.2, "g": {"sub": 0.0, "bass": 0.0, "low_mids": 0.0, "mids": 0.0, "presence": 0.0, "brilliance": +1.0}},
            {"name": "angry",   "v": 0.2, "a": 0.8, "g": {"sub": 0.0, "bass": +3.0, "low_mids": 0.0, "mids": 0.0, "presence": 0.0, "brilliance": +2.0}},
        ]

        if valence is None or arousal is None:
            v, a = anchors.get((emotion or "neutral").lower(), (0.5, 0.5))
        else:
            v = float(valence)
            a = float(arousal)

        # Primary/secondary fusion by VA blending when confidence is low
        try:
            conf = 0.7 if confidence is None else float(np.clip(confidence, 0.0, 1.0))
        except Exception:
            conf = 0.7
        if secondary:
            sec_v, sec_a = anchors.get(str(secondary).lower(), (v, a))
            # Blend weight grows as confidence drops; max 0.35 pull toward secondary
            w = 0.35 * (1.0 - conf)
            v = float(np.clip((1 - w) * v + w * sec_v, 0.0, 1.0))
            a = float(np.clip((1 - w) * a + w * sec_a, 0.0, 1.0))

        # Interpolate per-band gains across VA LUT (inverse-distance weighting)
        import math
        weights = []
        for node in va_lut:
            dv = v - node["v"]
            da = a - node["a"]
            d = math.hypot(dv, da)
            w = 1e9 if d == 0 else 1.0 / (d + 1e-6)
            weights.append((w, node["g"]))
        bands = {b["name"]: 0.0 for b in eq_bands}
        W = sum(w for w, _ in weights)
        for w, g in weights:
            for k in bands.keys():
                bands[k] += (g.get(k, 0.0) * w) / (W + 1e-9)

        # Scale by aggression and confidence (confidence-aware scaling)
        scale = float(aggression) * float(np.clip(conf, 0.4, 1.0))
        gains = {k: bands[k] * scale for k in bands}

        # Smooth adjacent bands
        order = [b["name"] for b in eq_bands]
        smoothed = gains.copy()
        for i in range(1, len(order) - 1):
            k = order[i]
            km1 = order[i - 1]
            kp1 = order[i + 1]
            smoothed[k] = (gains[k] * 0.6) + (gains[km1] * 0.2) + (gains[kp1] * 0.2)
        gains = smoothed

        # Content-aware limits: adapt highs/bass caps using arousal/valence proxies
        # Higher arousal -> lower highs cap to avoid harshness; low arousal -> allow a bit more air
        hi_cap = eq_limits["hi_total_boost_cap_db"]
        if a >= 0.7:
            hi_cap = min(hi_cap, 4.0)
        elif a <= 0.3:
            hi_cap = max(hi_cap, 7.0)

        # Low valence -> prefer warmth over excessive presence/air; high valence -> restrain bass
        bass_cap = eq_limits["boost_db_max"]
        if v <= 0.3:
            bass_cap = min(bass_cap, 5.0)
        elif v >= 0.7:
            bass_cap = min(bass_cap, 3.5)

        # Limits and high/bass caps
        for k, g in gains.items():
            if g >= 0:
                if k in ("sub", "bass"):
                    gains[k] = min(g, bass_cap)
                else:
                    gains[k] = min(g, eq_limits["boost_db_max"])
            else:
                gains[k] = max(g, -eq_limits["cut_db_max"])
        hi_sum = max(0.0, gains.get("presence", 0.0)) + max(0.0, gains.get("brilliance", 0.0))
        if hi_sum > hi_cap:
            factor = hi_cap / (hi_sum + 1e-9)
            gains["presence"] *= factor
            gains["brilliance"] *= factor

        # Soft total positive gain limiter to avoid over-brightness at high aggression
        pos_sum = sum(max(0.0, x) for x in gains.values())
        pos_cap = 12.0  # dB budget across all bands
        if pos_sum > pos_cap:
            k_soft = pos_cap / (pos_sum + 1e-9)
            for k in gains:
                if gains[k] > 0:
                    gains[k] *= k_soft

        # Build dense GraphicEQ by log-freq interpolation across bands
        frequencies = [
            20, 21, 22, 23, 24, 26, 27, 29, 30, 32, 34, 36, 38, 40, 43, 45, 48, 50, 53, 56, 59, 63, 66, 70, 74, 78, 83, 87, 92, 97,
            103, 109, 115, 121, 128, 136, 143, 151, 160, 169, 178, 188, 199, 210, 222, 235, 248, 262, 277, 292, 309, 326, 345, 364,
            385, 406, 429, 453, 479, 506, 534, 565, 596, 630, 665, 703, 743, 784, 829, 875, 924, 977, 1032, 1090, 1151, 1216, 1284,
            1357, 1433, 1514, 1599, 1689, 1784, 1885, 1991, 2103, 2221, 2347, 2479, 2618, 2766, 2921, 3086, 3260, 3443, 3637, 3842,
            4058, 4287, 4528, 4783, 5052, 5337, 5637, 5955, 6290, 6644, 7018, 7414, 7831, 8272, 8738, 9230, 9749, 10298, 10878, 11490,
            12137, 12821, 13543, 14305, 15110, 15961, 16860, 17809, 18812, 19871
        ]
        pts_x = [b["fc"] for b in eq_bands]
        pts_y = [gains[b["name"]] for b in eq_bands]
        lx = np.log10(np.array(pts_x))
        ly = np.array(pts_y)
        fx = np.log10(np.array(frequencies))
        eq_env = np.interp(fx, lx, ly, left=ly[0], right=ly[-1])
        wavelet_eq = "GraphicEQ: " + "; ".join([f"{f} {g:.2f}" for f, g in zip(frequencies, eq_env.tolist())])

        # Parametric lines
        lines = []
        for b in eq_bands:
            g = gains[b["name"]]
            if abs(g) < 0.25:
                continue
            q = min(max(b["q"], eq_limits["q_min"]), eq_limits["q_max"])
            lines.append(f"{b['type'].title()} {int(b['fc'])} Hz: {g:+.1f} dB, Q={q:.2f}")
        parametric_eq = "\n".join(lines) if lines else "Flat"

        description = {
            "happy": "Bright with enhanced presence",
            "sad": "Warm with softened highs",
            "calm": "Smooth and relaxed",
            "neutral": "Flat reference",
            "angry": "Tense with controlled highs",
        }.get((emotion or "neutral").lower(), "Flat reference")

        return {"wavelet": wavelet_eq, "parametric": parametric_eq, "description": description}

    def generate_spectrogram(self, audio_path: str) -> Optional[str]:
        """Generate and save a spectrogram image for the given audio.

        Returns the temporary PNG path on success, or None on failure.
        """
        try:
            y, sr = librosa.load(audio_path, sr=SR, mono=True)
            # Compute mel spectrogram for nicer look
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=96, hop_length=HOP_LENGTH, fmin=20, fmax=sr//2)
            S_db = librosa.power_to_db(S, ref=np.max)
            fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
            img = librosa.display.specshow(S_db, sr=sr, hop_length=HOP_LENGTH, x_axis='time', y_axis='mel', ax=ax)
            ax.set_title('Mel Spectrogram')
            fig.colorbar(img, ax=ax, format='%+2.0f dB')
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                fig.tight_layout()
                fig.savefig(tmp.name, bbox_inches='tight')
                path = tmp.name
            plt.close(fig)
            return path
        except Exception as e:
            print(f"Error generating spectrogram: {e}")
            return None

    def apply_eq_to_audio(
        self,
        audio_path: str,
        emotion: str,
        aggression: float = 0.5,
        *,
        confidence: Optional[float] = None,
        warmth: float = 0.0,
        presence: float = 0.0,
        air: float = 0.0,
        hq_linear: bool = False,
    ) -> Tuple[bytes, Dict]:
        """Apply a lightweight EQ and return PCM WAV bytes and simple stats.

        This is a pragmatic implementation to maintain API parity. It re-encodes
        input to PCM 16-bit WAV to avoid 'unsupported encoding' errors and applies
        gentle shelves/peaks based on emotion and controls.
        """
        try:
            y, sr = librosa.load(audio_path, sr=SR, mono=True)
            # Normalize to -1..1 range (librosa already does float32)
            y = np.clip(y, -1.0, 1.0)

            # Very lightweight EQ using IIR biquads (scipy.signal)
            def biquad_peaking(x, fc, q, gain_db, fs):
                # Convert to biquad using RBJ Audio Eq Cookbook
                A = 10**(gain_db/40)
                w0 = 2*np.pi*fc/fs
                alpha = np.sin(w0)/(2*q)
                b0 = 1 + alpha*A
                b1 = -2*np.cos(w0)
                b2 = 1 - alpha*A
                a0 = 1 + alpha/A
                a1 = -2*np.cos(w0)
                a2 = 1 - alpha/A
                b = np.array([b0, b1, b2]) / a0
                a = np.array([1.0, a1/a0, a2/a0])
                return signal.lfilter(b, a, x)

            def shelf_low(x, fc, gain_db, fs, high=False):
                # Simple first-order shelving via bilinear transform
                K = np.tan(np.pi*fc/fs)
                V0 = 10**(gain_db/20)
                if high:
                    # high-shelf
                    b0 = V0*(K**2) + np.sqrt(2*V0)*K + 1
                    b1 = 2*(V0*(K**2) - 1)
                    b2 = V0*(K**2) - np.sqrt(2*V0)*K + 1
                    a0 = (K**2) + np.sqrt(2)*K + 1
                    a1 = 2*((K**2) - 1)
                    a2 = (K**2) - np.sqrt(2)*K + 1
                else:
                    # low-shelf
                    b0 = V0*(K**2) + np.sqrt(2*V0)*K + 1
                    b1 = 2*(V0*(K**2) - 1)
                    b2 = V0*(K**2) - np.sqrt(2*V0)*K + 1
                    a0 = (K**2) + np.sqrt(2)*K + 1
                    a1 = 2*((K**2) - 1)
                    a2 = (K**2) - np.sqrt(2)*K + 1
                b = np.array([b0, b1, b2]) / a0
                a = np.array([1.0, a1/a0, a2/a0])
                return signal.lfilter(b, a, x)

            # Emotion-driven gentle EQ targets
            emo = (emotion or "neutral").lower()
            g_sub = g_bass = g_mids = g_hi = 0.0
            if emo == "happy":
                g_hi = +3.0
                g_mids = +1.0
            elif emo == "sad":
                g_bass = +1.5
                g_mids = -1.0
            elif emo == "angry":
                g_bass = +3.0
                g_hi = +2.0
            elif emo == "calm":
                g_hi = +1.0
            # Scale by aggression (0..1.5 clamp)
            alpha = float(np.clip(aggression, 0.0, 1.5))
            g_sub *= alpha
            g_bass *= alpha
            g_mids *= alpha
            g_hi *= alpha
            # User tonal trims
            g_bass += float(warmth) * 2.0
            g_mids += float(presence) * 1.5
            g_hi += float(air) * 2.0

            # HQ linear-phase path: render full GraphicEQ as FIR and convolve
            if hq_linear:
                # 1) Get GraphicEQ from the same generator used for visualization
                eq = self.generate_eq_curves(emotion, aggression)
                wave = eq.get("wavelet", "GraphicEQ: 20 0; 20000 0")
                # 2) Parse points
                def parse_wavelet(s: str):
                    s = s.replace("GraphicEQ:", "").strip()
                    pts = []
                    for part in s.split(";"):
                        part = part.strip()
                        if not part:
                            continue
                        bits = part.split()
                        if len(bits) >= 2:
                            try:
                                pts.append((float(bits[0]), float(bits[1])))
                            except Exception:
                                pass
                    # ensure sorted by freq
                    pts.sort(key=lambda t: t[0])
                    return pts
                pts = parse_wavelet(wave)
                if len(pts) < 4:
                    # Fallback to IIR if malformed
                    x = y
                    if abs(g_bass) > 0.05:
                        x = shelf_low(x, 120.0, g_bass, SR, high=False)
                    if abs(g_mids) > 0.05:
                        x = biquad_peaking(x, 2000.0, q=1.0, gain_db=g_mids, fs=SR)
                    if abs(g_hi) > 0.05:
                        x = shelf_low(x, 8000.0, g_hi, SR, high=True)
                else:
                    freqs = np.array([p[0] for p in pts], dtype=float)
                    gains_db = np.array([p[1] for p in pts], dtype=float)
                    # Apply user tonal trims directly to the curve
                    # Warmth: < 150 Hz, Presence: 1k-4k, Air: > 8k
                    warmth_db = float(warmth) * 2.0
                    presence_db = float(presence) * 1.5
                    air_db = float(air) * 2.0
                    mask_w = freqs < 150.0
                    mask_p = (freqs >= 1000.0) & (freqs <= 4000.0)
                    mask_a = freqs > 8000.0
                    gains_db = gains_db + mask_w*warmth_db + mask_p*presence_db + mask_a*air_db
                    # 3) Convert dB to linear magnitudes
                    mags = 10**(gains_db / 20.0)
                    # 4) Design FIR via frequency sampling (firwin2 expects [0,1] normalized freq)
                    # Build a moderately long filter for good resolution
                    numtaps = 4096
                    # Clamp Nyquist and zero
                    f_n = np.clip(freqs / (SR/2.0), 0.0, 1.0)
                    # Ensure endpoints 0 and 1 present
                    if f_n[0] > 0:
                        f_n = np.insert(f_n, 0, 0.0)
                        mags = np.insert(mags, 0, mags[0])
                    if f_n[-1] < 1.0:
                        f_n = np.append(f_n, 1.0)
                        mags = np.append(mags, mags[-1])
                    try:
                        taps = signal.firwin2(numtaps, f_n, mags, window="hann")
                        x = signal.fftconvolve(y, taps, mode="same")
                    except Exception:
                        # Safety fallback to IIR path
                        x = y
                        if abs(g_bass) > 0.05:
                            x = shelf_low(x, 120.0, g_bass, SR, high=False)
                        if abs(g_mids) > 0.05:
                            x = biquad_peaking(x, 2000.0, q=1.0, gain_db=g_mids, fs=SR)
                        if abs(g_hi) > 0.05:
                            x = shelf_low(x, 8000.0, g_hi, SR, high=True)
            else:
                # Fast IIR approximation
                x = y
                if abs(g_bass) > 0.05:
                    x = shelf_low(x, 120.0, g_bass, SR, high=False)
                if abs(g_mids) > 0.05:
                    x = biquad_peaking(x, 2000.0, q=1.0, gain_db=g_mids, fs=SR)
                if abs(g_hi) > 0.05:
                    x = shelf_low(x, 8000.0, g_hi, SR, high=True)
            x = np.clip(x, -1.0, 1.0)

            # Optional simple loudness polish (keep headroom)
            peak = np.max(np.abs(x)) + 1e-9
            if peak > 0.99:
                x = x / peak * 0.99

            # Encode to PCM 16-bit WAV bytes
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                sf.write(tmp.name, x, SR, subtype="PCM_16")
                wav_path = tmp.name
            with open(wav_path, "rb") as f:
                audio_bytes = f.read()
            try:
                os.unlink(wav_path)
            except Exception:
                pass

            stats = {}
            return audio_bytes, stats
        except Exception as e:
            print(f"Error applying EQ: {e}")
            return b"", {}

if __name__ == "__main__":
    # Example usage
    model_path = "best_val_loss.pth"
    labels_path = "labels.json"
    scaler_mean_path = "scaler_mean.npy"
    scaler_scale_path = "scaler_scale.npy"
    
    predictor = MusicEmotionPredictor(
        model_path=model_path,
        labels_path=labels_path,
        scaler_mean_path=scaler_mean_path,
        scaler_scale_path=scaler_scale_path
    )
    
    # Test prediction
    test_audio = "sample_audio.mp3"  # Replace with actual audio file
    if os.path.exists(test_audio):
        result = predictor.predict_emotion(test_audio)
        print("\nPrediction Result:")
        print(f"Predicted Emotion: {result['emotion']} (confidence: {result['confidence']:.2f})")
        print("All Probabilities:")
        for emo, prob in result['probabilities'].items():
            print(f"  {emo}: {prob:.4f}")
    else:
        print(f"Test audio file not found: {test_audio}")
