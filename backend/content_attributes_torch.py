#!/usr/bin/env python3
# backend/content_attributes_torch.py
"""
Content attributes via PANNs (PyTorch, AudioSet tags):
- vocals_score: vocals/singing/speech presence
- sibilance_score: cymbals/hi-hat/bright high-frequency content
- bass_score: bass/kick/low-frequency content

PyTorch-only. No TensorFlow required.

Install:
  python3.11 -m pip install --user --break-system-packages panns-inference torch torchaudio librosa soundfile numpy

Usage:
  extractor = ContentAttributeExtractorTorch()  # loads PANNs model on first use (CPU by default)
  attrs = extractor.extract("path/to/audio.wav")
  # -> {
  #   "vocals_score": float,
  #   "sibilance_score": float,
  #   "bass_score": float,
  #   "raw": {"top_tags": [...], "top_scores": [...], "sample_rate": 32000}
  # }
"""

import os
import numpy as np
import librosa
import soundfile as sf


def _aggregate_keywords(tag_scores: dict[str, float], keywords: list[str]) -> float:
    """Sum probabilities of tags whose name contains any keyword."""
    kws = [k.lower() for k in keywords]
    s = 0.0
    for name, score in tag_scores.items():
        lname = name.lower()
        if any(k in lname for k in kws):
            s += float(score)
    # Clip to [0,1] as a simple normalization cap (these are probabilities from a sigmoid head)
    return float(np.clip(s, 0.0, 1.0))


class ContentAttributeExtractorTorch:
    """
    Extracts robust content attributes using pretrained PANNs:
      - vocals_score: vocals/singing/speech indicators
      - sibilance_score: cymbal/hi-hat/bright-highs indicators
      - bass_score: bass/kick/low-frequency indicators
    """

    # Keyword sets mapped to AudioSet tag names used by PANNs labels
    VOCALS_KW = ["vocal", "singing", "singer", "speech", "choir", "vocal music", "narration", "rap"]
    SIBILANCE_KW = ["cymbal", "hi-hat", "ride cymbal", "crash cymbal", "sizzle", "shaker", "hihat"]
    BASS_KW = ["bass", "bass guitar", "bass drum", "kick drum", "sub-bass", "low frequency"]

    def __init__(self, device: str = "cpu"):
        """
        device: "cpu" or "cuda" (if available). CPU is fine for these lightweight attributes.
        Lazily imports panns_inference to avoid hard dependency failures at server import time.
        If PANNs assets are missing, extractor will return neutral attributes and not crash.
        """
        self.device = device
        # PANNs expects 32 kHz mono waveform
        self.target_sr = 32000
        self.model = None
        self.labels = None
        try:
            from panns_inference import AudioTagging, labels as PANN_LABELS  # noqa: F401
            # Initialize the AudioTagging model once (may create ~/panns_data)
            self.model = AudioTagging(checkpoint_path=None, device=device)
            self.labels = PANN_LABELS
        except Exception as e:
            # Defer failure: operate in neutral mode
            self.model = None
            self.labels = []
            self._init_error = str(e)

    def _load_resample_mono(self, audio_path: str) -> np.ndarray:
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        if sr != self.target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.target_sr)
        return y.astype(np.float32)

    def extract(self, audio_path: str) -> dict:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(audio_path)

        # If model failed to initialize, return neutral attributes
        if self.model is None:
            return {
                "vocals_score": 0.0,
                "sibilance_score": 0.0,
                "bass_score": 0.0,
                "raw": {"note": "panns_inference_unavailable", "error": getattr(self, "_init_error", None)},
            }

        y = self._load_resample_mono(audio_path)

        # PANNs AudioTagging API:
        # outputs = self.model.inference(y) -> dict with 'clipwise_output' (np.ndarray [527]) and 'labels' (list)
        with np.errstate(invalid="ignore"):
            outputs = self.model.inference(y)

        clip_scores = outputs["clipwise_output"]  # shape [527], float probabilities per AudioSet class
        labels = outputs.get("labels", self.labels)

        # Build a name->score mapping
        tag_scores = {labels[i]: float(clip_scores[i]) for i in range(len(labels))}

        # Aggregate attributes
        vocals = _aggregate_keywords(tag_scores, self.VOCALS_KW)
        sibilance = _aggregate_keywords(tag_scores, self.SIBILANCE_KW)
        bass = _aggregate_keywords(tag_scores, self.BASS_KW)

        # Prepare top-5 tags for simple debugging / UI
        top_idx = np.argsort(clip_scores)[-5:][::-1]
        top_tags = [labels[i] for i in top_idx]
        top_vals = [float(clip_scores[i]) for i in top_idx]

        return {
            "vocals_score": vocals,
            "sibilance_score": sibilance,
            "bass_score": bass,
            "raw": {
                "top_tags": top_tags,
                "top_scores": top_vals,
                "sample_rate": self.target_sr,
            },
        }


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", help="Path to audio file")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()

    ext = ContentAttributeExtractorTorch(device=args.device)
    out = ext.extract(args.audio)
    print(json.dumps(out, indent=2))
