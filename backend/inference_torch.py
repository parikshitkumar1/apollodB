# apollodB - PyTorch Music Emotion Inference
# New PyTorch-only pipeline with content attributes

from __future__ import annotations

import os
import io
import json
from typing import Dict, List, Tuple, Optional

import numpy as np
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
try:
    # Allowlist numpy reconstruct used in some older checkpoints
    from numpy.core.multiarray import _reconstruct as _np_reconstruct  # type: ignore
    import torch.serialization as _ts
    if hasattr(_ts, "add_safe_globals"):
        _ts.add_safe_globals([_np_reconstruct])
except Exception:
    pass

from backend.content_attributes_torch import ContentAttributeExtractorTorch

# Audio parameters (defaults; can be overridden by artifact config)
SR = 22050
DURATION = 30
N_MELS = 96
HOP_LENGTH = 512
FMIN = 20
FMAX = 11025
MAX_LEN = 1300

# Emotions (UI categories). Primary runtime is continuous VA; categories are derived.
EMOTIONS = ["neutral", "happy", "sad", "calm", "angry"]


def _derive_category_from_va(valence: float, arousal: float) -> str:
    """Map continuous VA in [0,1] to 5 coarse categories for UI only (includes angry)."""
    v = float(np.clip(valence, 0.0, 1.0))
    a = float(np.clip(arousal, 0.0, 1.0))
    if a > 0.65 and v < 0.4:
        return "angry"
    if v >= 0.6 and a >= 0.55:
        return "happy"
    if v < 0.4 and a < 0.5:
        return "sad"
    if a < 0.35 and v >= 0.55:
        return "calm"
    return "neutral"


class _MelExtractor:
    def __init__(self, sr=SR, n_mels=N_MELS, hop_length=HOP_LENGTH, duration=DURATION, max_len=MAX_LEN, fmin=FMIN, fmax=FMAX):
        self.sr = sr
        self.n_mels = n_mels
        self.hop = hop_length
        self.duration = duration
        self.max_len = max_len
        self.fmin = fmin
        self.fmax = fmax

    def __call__(self, audio_path: str) -> np.ndarray:
        y, sr = librosa.load(audio_path, sr=self.sr, mono=True)
        # pad/truncate to fixed duration
        target_len = self.sr * self.duration
        if len(y) > target_len:
            y = y[:target_len]
        else:
            y = np.pad(y, (0, target_len - len(y)))
        S = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=self.n_mels, hop_length=self.hop, fmin=self.fmin, fmax=self.fmax)
        S_db = librosa.power_to_db(S, ref=np.max).T  # (time, mels)
        # pad/truncate time dim
        if S_db.shape[0] < self.max_len:
            pad = self.max_len - S_db.shape[0]
            S_db = np.pad(S_db, ((0, pad), (0, 0)))
        else:
            S_db = S_db[:self.max_len]
        return S_db.astype(np.float32)


# --------------------------- Minimal model used in training ---------------------------

class _ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=(5,5), s=(2,2), p=(2,2)):
        super().__init__()
        self.c = nn.Conv2d(in_ch, out_ch, k, s, p)
        self.b = nn.BatchNorm2d(out_ch)
        self.a = nn.GELU()
    def forward(self, x):
        return self.a(self.b(self.c(x)))


class _TransformerEncoder(nn.Module):
    def __init__(self, d, h, m, p=0.2):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, h, p, batch_first=True)
        self.d1 = nn.Dropout(p)
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, m), nn.GELU(), nn.Dropout(p), nn.Linear(m, d))
        self.d2 = nn.Dropout(p)
    def forward(self, x):
        r = x
        x = self.n1(x)
        a,_ = self.att(x, x, x)
        x = r + self.d1(a)
        r = x
        x = self.n2(x)
        return r + self.d2(self.ff(x))


class _AttentiveStatsPooling(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.att = nn.Sequential(nn.Linear(d, d//2), nn.Tanh(), nn.Linear(d//2, 1))
    def forward(self, x):
        a = torch.softmax(self.att(x), dim=1)
        mean = (a * x).sum(1)
        var = (a * (x - mean.unsqueeze(1))**2).sum(1).clamp_min(1e-6)
        return torch.cat([mean, var.sqrt()], dim=1)


class EmotionVA(nn.Module):
    def __init__(self, n_mels=N_MELS, d=128, L=4, h=4, m=256, p=0.2):
        super().__init__()
        self.cnn = nn.Sequential(
            _ConvBlock(1, 16),
            _ConvBlock(16, 32, k=(3,3), s=(2,2), p=(1,1)),
            _ConvBlock(32, 64, k=(3,3), s=(2,2), p=(1,1))
        )
        self.proj = nn.Linear(64 * int(np.ceil(n_mels/8)), d)
        self.tr = nn.ModuleList([_TransformerEncoder(d, h, m, p) for _ in range(L)])
        self.pool = _AttentiveStatsPooling(d)
        self.head = nn.Linear(2*d, 2)
    def forward(self, x):  # x [B,T,F]
        x = x.unsqueeze(1)  # [B,1,T,F]
        x = self.cnn(x)    # [B,64,T',F']
        B, C, Tp, Fp = x.shape
        x = x.permute(0,2,1,3).contiguous().view(B, Tp, C*Fp)
        x = self.proj(x)
        for blk in self.tr:
            x = blk(x)
        x = self.pool(x)
        return self.head(x)


class MusicEmotionPredictor:
    """
    PyTorch continuous VA predictor + content attributes.

    Exposes methods compatible with the existing FastAPI backend:
      - predict_emotion(audio_path)
      - generate_eq_curves(emotion, aggression)
      - apply_eq_to_audio(...)
      - generate_spectrogram(audio_path)
    """

    def __init__(self, artifact_path: str = None, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Try default weights path if not provided
        self.artifact_path = artifact_path or os.environ.get("APOLLODB_VA_WEIGHTS", "weights/apollodb_va.pt")
        self.model: Optional[nn.Module] = None
        self.scaler_mean: Optional[np.ndarray] = None
        self.scaler_scale: Optional[np.ndarray] = None
        self.config: Dict = {}
        self.attr_extractor = ContentAttributeExtractorTorch(device="cpu")  # lightweight; keep on CPU

        # Load model/scaler/config first
        self._load_artifact()

        # Build mel extractor using config overrides if available
        sr = int(self.config.get("sr", SR)) if isinstance(self.config, dict) else SR
        n_mels = int(self.config.get("n_mels", N_MELS)) if isinstance(self.config, dict) else N_MELS
        hop = int(self.config.get("hop_length", HOP_LENGTH)) if isinstance(self.config, dict) else HOP_LENGTH
        dur = int(self.config.get("duration", DURATION)) if isinstance(self.config, dict) else DURATION
        max_len = int(self.config.get("max_len", MAX_LEN)) if isinstance(self.config, dict) else MAX_LEN
        fmin = int(self.config.get("fmin", FMIN)) if isinstance(self.config, dict) else FMIN
        fmax = int(self.config.get("fmax", FMAX)) if isinstance(self.config, dict) else FMAX
        self.mel = _MelExtractor(sr=sr, n_mels=n_mels, hop_length=hop, duration=dur, max_len=max_len, fmin=fmin, fmax=fmax)

        # Fixed psychoacoustic bands and default Q
        self.eq_bands = [
            {"name": "sub", "type": "lowshelf", "fc": 45.0, "q": 0.7},
            {"name": "bass", "type": "peaking", "fc": 90.0, "q": 0.9},
            {"name": "low_mids", "type": "peaking", "fc": 250.0, "q": 1.0},
            {"name": "mids", "type": "peaking", "fc": 800.0, "q": 1.0},
            {"name": "presence", "type": "peaking", "fc": 3000.0, "q": 1.1},
            {"name": "brilliance", "type": "highshelf", "fc": 12000.0, "q": 0.7},
        ]
        self.va_lut = [
            {"v": 0.2, "a": 0.2, "g": {"sub": 0.0, "bass": -0.5, "low_mids": +0.5, "mids": +0.5, "presence": -0.5, "brilliance": -0.5}},
            {"v": 0.8, "a": 0.2, "g": {"sub": +1.0, "bass": +1.5, "low_mids": -0.5, "mids": -0.5, "presence": +0.5, "brilliance": +0.5}},
            {"v": 0.2, "a": 0.8, "g": {"sub": -0.5, "bass": 0.0, "low_mids": -0.5, "mids": +0.5, "presence": +1.5, "brilliance": +1.0}},
            {"v": 0.8, "a": 0.8, "g": {"sub": +0.5, "bass": +1.0, "low_mids": -0.5, "mids": 0.0, "presence": +1.0, "brilliance": +1.0}},
            {"v": 0.5, "a": 0.5, "g": {"sub": 0.0, "bass": 0.0, "low_mids": 0.0, "mids": 0.0, "presence": 0.0, "brilliance": 0.0}},
        ]
        self.eq_limits = {"boost_db_max": 6.0, "cut_db_max": 8.0, "q_min": 0.5, "q_max": 1.6, "hi_total_boost_cap_db": 3.0}

    def _load_artifact(self) -> None:
        """Load TorchScript or state_dict artifact. Requires PyTorch-only runtime."""
        if not self.artifact_path or not os.path.exists(self.artifact_path):
            # Defer hard failure to predict time with helpful message
            return
        try:
            # Try TorchScript first
            self.model = torch.jit.load(self.artifact_path, map_location=self.device)
            self.model.eval()
            # Try to read auxiliary dict saved next to scripted model (optional)
            aux_path = os.path.splitext(self.artifact_path)[0] + ".aux.json"
            if os.path.exists(aux_path):
                with open(aux_path, "r") as f:
                    aux = json.load(f)
                self.scaler_mean = np.array(aux.get("scaler_mean", []), dtype=np.float32) if aux.get("scaler_mean") is not None else None
                self.scaler_scale = np.array(aux.get("scaler_scale", []), dtype=np.float32) if aux.get("scaler_scale") is not None else None
                self.config = aux.get("config", {})
            return
        except Exception:
            pass
        # Fallback: load as torch pickle (expects a dict)
        try:
            blob = torch.load(self.artifact_path, map_location=self.device, weights_only=False)
            if isinstance(blob, dict):
                self.config = blob.get("config", {})
                sm = blob.get("scaler_mean")
                ss = blob.get("scaler_scale")
                if sm is not None and ss is not None:
                    self.scaler_mean = np.array(sm, dtype=np.float32)
                    self.scaler_scale = np.array(ss, dtype=np.float32)
                # If a scripted module is embedded
                model_obj = blob.get("scripted_model") or blob.get("model_scripted")
                if isinstance(model_obj, torch.jit.ScriptModule) or isinstance(model_obj, torch.jit.RecursiveScriptModule):
                    self.model = model_obj
                    self.model.eval()
                    return
                # If state dict is present under common keys, reconstruct model
                state = blob.get("model_state") or blob.get("model_state_dict") or blob.get("state_dict")
                if state is not None:
                    # Rebuild model with n_mels from config when present
                    n_mels = int(self.config.get("n_mels", N_MELS)) if isinstance(self.config, dict) else N_MELS
                    mdl = EmotionVA(n_mels=n_mels)
                    mdl.load_state_dict(state, strict=True)
                    self.model = mdl.to(self.device).eval()
                    return
        except Exception as e:
            raise RuntimeError(f"Failed to load VA model artifact '{self.artifact_path}': {e}")

    def _normalize_features(self, F: np.ndarray) -> np.ndarray:
        if self.scaler_mean is None or self.scaler_scale is None:
            return F
        return (F - self.scaler_mean) / (self.scaler_scale + 1e-8)

    # --------------------------- Core API ---------------------------
    def predict_emotion(self, audio_path: str) -> Dict:
        """Predict continuous VA and derive a UI category. Includes content attributes.
        Returns keys: valence, arousal, primary_emotion, secondary_emotion, confidence, emotion_probabilities (derived),
                      content_attributes {vocals_score, sibilance_score, bass_score}
        """
        if self.model is None:
            raise RuntimeError(
                "VA model is not loaded. Set APOLLODB_VA_WEIGHTS or pass artifact_path to MusicEmotionPredictor."
            )
        # 1) Features
        F = self.mel(audio_path)  # (T, M)
        Fn = self._normalize_features(F)
        x = torch.from_numpy(Fn).unsqueeze(0).to(self.device)  # [1, T, M]
        # Some models expect [B, 1, T, M]; try to adapt automatically
        try:
            with torch.no_grad():
                out = self.model(x)
        except Exception:
            x2 = x.unsqueeze(1)  # [1, 1, T, M]
            with torch.no_grad():
                out = self.model(x2)
        if isinstance(out, (tuple, list)):
            out = out[0]
        va = out.detach().cpu().float().numpy().reshape(-1)
        if va.shape[0] >= 2:
            v, a = float(va[0]), float(va[1])
        else:
            raise RuntimeError("Model output does not contain 2 values for (valence, arousal)")
        # Clamp to [0,1] if needed
        v = float(np.clip(v, 0.0, 1.0))
        a = float(np.clip(a, 0.0, 1.0))

        # 2) Derived emotions (for UI)
        primary = _derive_category_from_va(v, a)
        # second-best heuristic by flipping axis slightly
        alt = _derive_category_from_va(min(1.0, v * 0.9 + 0.05), min(1.0, a * 0.9 + 0.05))
        secondary = alt if alt != primary else "neutral"
        probs = {e: 0.0 for e in EMOTIONS}
        probs[primary] = 0.7
        if secondary in probs:
            probs[secondary] = 0.25
        probs["neutral"] = max(probs["neutral"], 0.05)
        confidence = max(probs.values())

        # 3) Content attributes (PyTorch PANNs)
        attrs = {}
        try:
            attrs = self.attr_extractor.extract(audio_path)
        except Exception:
            attrs = {"vocals_score": 0.0, "sibilance_score": 0.0, "bass_score": 0.0, "raw": {}}

        return {
            "valence": v,
            "arousal": a,
            "primary_emotion": primary,
            "secondary_emotion": secondary,
            "emotion_probabilities": probs,
            "confidence": float(confidence),
            "content_attributes": {
                "vocals_score": float(attrs.get("vocals_score", 0.0)),
                "sibilance_score": float(attrs.get("sibilance_score", 0.0)),
                "bass_score": float(attrs.get("bass_score", 0.0)),
            },
        }

    def emotion_to_va(self, emotion: str) -> Tuple[float, float]:
        m = {
            "neutral": (0.5, 0.5),
            "happy": (0.8, 0.8),
            "sad": (0.2, 0.2),
            "calm": (0.8, 0.2),
            "angry": (0.2, 0.8),
        }
        return m.get(emotion, (0.5, 0.5))

    def generate_eq_curves(self, emotion: str, aggression: float = 0.5) -> Dict[str, str]:
        """Generate EQ curves via VA LUT with safe constraints."""
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
            bands = {b["name"]: 0.0 for b in self.eq_bands}
            W = sum(w for w, _ in weights)
            for w, g in weights:
                for k in bands.keys():
                    bands[k] += (g.get(k, 0.0) * w) / (W + 1e-9)
            return bands

        base_gains = interp_va(v, a)

        # Confidence unknown here; assume mid
        conf = 0.7
        scale = max(0.3, min(1.0, conf)) * float(aggression)
        gains = {k: base_gains[k] * scale for k in base_gains}

        # Smooth between adjacent bands
        order = [b["name"] for b in self.eq_bands]
        smoothed = gains.copy()
        for i in range(1, len(order) - 1):
            k = order[i]
            km1 = order[i - 1]
            kp1 = order[i + 1]
            smoothed[k] = (gains[k] * 0.6) + (gains[km1] * 0.2) + (gains[kp1] * 0.2)
        gains = smoothed

        # Limits and high-band cap
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

        # Build GraphicEQ line by log-freq interpolation
        frequencies = [
            20, 21, 22, 23, 24, 26, 27, 29, 30, 32, 34, 36, 38, 40, 43, 45, 48, 50, 53, 56, 59, 63, 66, 70, 74, 78, 83, 87, 92, 97,
            103, 109, 115, 121, 128, 136, 143, 151, 160, 169, 178, 188, 199, 210, 222, 235, 248, 262, 277, 292, 309, 326, 345, 364,
            385, 406, 429, 453, 479, 506, 534, 565, 596, 630, 665, 703, 743, 784, 829, 875, 924, 977, 1032, 1090, 1151, 1216, 1284,
            1357, 1433, 1514, 1599, 1689, 1784, 1885, 1991, 2103, 2221, 2347, 2479, 2618, 2766, 2921, 3086, 3260, 3443, 3637, 3842,
            4058, 4287, 4528, 4783, 5052, 5337, 5637, 5955, 6290, 6644, 7018, 7414, 7831, 8272, 8738, 9230, 9749, 10298, 10878, 11490,
            12137, 12821, 13543, 14305, 15110, 15961, 16860, 17809, 18812, 19871
        ]
        pts_x = [b["fc"] for b in self.eq_bands]
        pts_y = [gains[b["name"]] for b in self.eq_bands]
        lx = np.log10(np.array(pts_x))
        ly = np.array(pts_y)
        fx = np.log10(np.array(frequencies))
        eq_env = np.interp(fx, lx, ly, left=ly[0], right=ly[-1])
        wavelet_eq = "GraphicEQ: " + "; ".join([f"{f} {g:.2f}" for f, g in zip(frequencies, eq_env.tolist())])

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
            "neutral": "Flat reference",
            "angry": "Tense with controlled highs",
        }.get(emotion, "Flat reference")

        return {"wavelet": wavelet_eq, "parametric": parametric_eq, "description": description}

    def generate_spectrogram(self, audio_path: str, save_path: str = None) -> Optional[str]:
        try:
            import librosa.display
            y, sr = librosa.load(audio_path, sr=SR, mono=True)
            if len(y) > SR * DURATION:
                y = y[:SR * DURATION]
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH, fmin=FMIN, fmax=FMAX)
            S_db = librosa.power_to_db(S, ref=np.max)
            plt.figure(figsize=(12, 6))
            plt.style.use('dark_background')
            librosa.display.specshow(S_db, sr=sr, hop_length=HOP_LENGTH, x_axis='time', y_axis='mel', 
                                  cmap='plasma', fmin=FMIN, fmax=FMAX)
            plt.colorbar(format='%+2.0f dB', label='Power (dB)')
            plt.title('Mel Spectrogram', color='white', fontsize=14, fontweight='bold')
            plt.xlabel('Time (s)', color='white')
            plt.ylabel('Frequency (Hz)', color='white')
            plt.tight_layout()
            plt.gca().set_facecolor('black')
            plt.gcf().patch.set_facecolor('black')
            if save_path:
                plt.savefig(save_path, facecolor='black', edgecolor='none', dpi=150, bbox_inches='tight')
                plt.close()
                return save_path
            else:
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
        """Apply EQ with VA LUT and safety/fidelity polish. Based on existing backend behavior."""
        try:
            import scipy.signal
            import soundfile as sf
            import tempfile
            import os

            # First try to read the file with soundfile
            try:
                info = sf.info(audio_path)
                orig_sr = info.samplerate
                orig_channels = info.channels
                orig_subtype = info.subtype or 'PCM_24'
                y, sr = sf.read(audio_path, always_2d=True)
            except Exception as e:
                # If soundfile fails, try loading with librosa and converting
                try:
                    import librosa
                    y, sr = librosa.load(audio_path, sr=None, mono=False)
                    if y.ndim == 1:
                        y = y.reshape(1, -1)  # Convert to 2D (channels, samples)
                    orig_sr = sr
                    orig_channels = y.shape[0]
                    orig_subtype = 'PCM_24'  # Default to PCM_24 for converted audio
                except Exception as lib_e:
                    raise RuntimeError(f"Failed to read audio file with both soundfile and librosa: {str(e)}, {str(lib_e)}")

            if sr != orig_sr:
                orig_sr = sr

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
            conf = 0.7 if (confidence is None) else float(np.clip(confidence, 0.0, 1.0))
            scale = max(0.3, min(1.0, conf)) * float(aggression)

            # Attribute-aware modifiers (vocals/sibilance/bass)
            try:
                attrs = self.attr_extractor.extract(audio_path)
            except Exception:
                attrs = {"vocals_score": 0.0, "sibilance_score": 0.0, "bass_score": 0.0}
            mods = {k: 0.0 for k in base_gains}
            if attrs.get("sibilance_score", 0.0) > 0.4:
                mods["presence"] -= min(1.5, (attrs["sibilance_score"] - 0.4) * 4.0)
                mods["brilliance"] -= min(2.0, (attrs["sibilance_score"] - 0.4) * 5.0)
            if attrs.get("bass_score", 0.0) > 0.5:
                mods["bass"] += min(1.5, (attrs["bass_score"] - 0.5) * 3.0)
                mods["low_mids"] -= 0.3

            # Macro user nudges
            mods["low_mids"] += float(warmth) * 1.5
            mods["presence"] += float(presence) * 1.5
            mods["brilliance"] += float(air) * 1.8

            gains = {k: (base_gains[k] + mods.get(k, 0.0)) * scale for k in base_gains}

            # Enforce limits
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

            # Design simple IIR filters per band
            def design_peaking(fc, q, gain_db, fs):
                A = 10 ** (gain_db / 40)
                w0 = 2 * np.pi * fc / fs
                alpha = np.sin(w0) / (2 * q)
                b0 = 1 + alpha * A
                b1 = -2 * np.cos(w0)
                b2 = 1 - alpha * A
                a0 = 1 + alpha / A
                a1 = -2 * np.cos(w0)
                a2 = 1 - alpha / A
                b = np.array([b0, b1, b2]) / a0
                a = np.array([1.0, a1 / a0, a2 / a0])
                return b, a

            def design_shelf(fc, gain_db, fs, high=False, q=0.707):
                A = 10 ** (gain_db / 40)
                w0 = 2 * np.pi * fc / fs
                alpha = np.sin(w0) / (2 * q)
                cosw0 = np.cos(w0)
                if high:
                    b0 = A * ((A + 1) + (A - 1) * cosw0 + 2 * np.sqrt(A) * alpha)
                    b1 = -2 * A * ((A - 1) + (A + 1) * cosw0)
                    b2 = A * ((A + 1) + (A - 1) * cosw0 - 2 * np.sqrt(A) * alpha)
                    a0 = (A + 1) - (A - 1) * cosw0 + 2 * np.sqrt(A) * alpha
                    a1 = 2 * ((A - 1) - (A + 1) * cosw0)
                    a2 = (A + 1) - (A - 1) * cosw0 - 2 * np.sqrt(A) * alpha
                else:
                    b0 = A * ((A + 1) - (A - 1) * cosw0 + 2 * np.sqrt(A) * alpha)
                    b1 = 2 * A * ((A - 1) - (A + 1) * cosw0)
                    b2 = A * ((A + 1) - (A - 1) * cosw0 - 2 * np.sqrt(A) * alpha)
                    a0 = (A + 1) + (A - 1) * cosw0 + 2 * np.sqrt(A) * alpha
                    a1 = -2 * ((A - 1) + (A + 1) * cosw0)
                    a2 = (A + 1) + (A - 1) * cosw0 - 2 * np.sqrt(A) * alpha
                b = np.array([b0, b1, b2]) / a0
                a = np.array([1.0, a1 / a0, a2 / a0])
                return b, a

            filters = []
            for b in self.eq_bands:
                g = gains[b["name"]]
                if abs(g) < 0.25:
                    continue
                if b["type"] == "peaking":
                    ba = design_peaking(b["fc"], b["q"], g, orig_sr)
                elif b["type"] == "lowshelf":
                    ba = design_shelf(b["fc"], g, orig_sr, high=False, q=b["q"]) 
                elif b["type"] == "highshelf":
                    ba = design_shelf(b["fc"], g, orig_sr, high=True, q=b["q"]) 
                else:
                    continue
                filters.append(ba)

            # Apply filters channel-wise
            y_out = y.copy()
            for (b, a) in filters:
                y_out = scipy.signal.lfilter(b, a, y_out, axis=0)

            # Simple headroom management
            peak = np.max(np.abs(y_out)) + 1e-9
            preamp = 0.99 / peak if peak > 1.0 else 1.0
            y_out = (y_out * preamp).astype(np.float32)

            # Compute simple stats
            true_peak = 20 * np.log10(float(np.max(np.abs(y_out)) + 1e-9))
            try:
                import pyloudnorm as pyln
                meter = pyln.Meter(orig_sr)
                lufs = float(meter.integrated_loudness(y_out.mean(axis=1)))
            except Exception:
                lufs = None

            import tempfile
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            sf.write(tmp.name, y_out, samplerate=orig_sr, subtype=orig_subtype)
            with open(tmp.name, 'rb') as f:
                audio_bytes = f.read()
            # keep file for caller if needed; here we return bytes only
            return audio_bytes, {"true_peak_dbtp": true_peak, "lufs": lufs}

        except Exception as e:
            print(f"Error applying EQ: {e}")
            return None


# --------------------------- Batch helper ---------------------------

def predict_multiple_files(file_paths: List[str], predictor: MusicEmotionPredictor) -> Dict:
    """Analyze multiple files and return aggregate results similar to legacy API."""
    individual = []
    counts = {e: 0 for e in EMOTIONS}
    for p in file_paths:
        try:
            r = predictor.predict_emotion(p)
            individual.append(r)
            counts[r["primary_emotion"]] = counts.get(r["primary_emotion"], 0) + 1
        except Exception as e:
            individual.append({"error": str(e)})
    dominant = max(counts.items(), key=lambda kv: kv[1])[0] if counts else "neutral"
    return {
        "individual_results": individual,
        "dominant_emotion": dominant,
        "counts": counts,
    }
