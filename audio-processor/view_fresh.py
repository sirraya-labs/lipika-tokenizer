#!/usr/bin/env python3
"""
lipika_viewer.py — Lipika Audio Token Viewer
=============================================

Displays discrete audio token sequences for a Lipika model.

Two modes:
  1. --checkpoint PATH   Real trained model. Shows actual encoded tokens.
  2. (no checkpoint)     Diagnostic mode. Simulates what a healthy trained
                         model's token stream looks like, using the real
                         audio's mel spectrogram to drive code assignments
                         so the pattern is actually audio-correlated.

Usage:
  # Trained model (real tokens):
  python lipika_viewer.py --audio speech.wav --checkpoint ckpt_step00050000.pt

  # Diagnostic / fresh model (simulated realistic tokens):
  python lipika_viewer.py --audio speech.wav
"""

from __future__ import annotations

import sys
import math
import argparse
import warnings
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np

warnings.filterwarnings("ignore")

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("ERROR: torch not installed. pip install torch")
    sys.exit(1)

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


# ── ANSI colours ──────────────────────────────────────────────────────────────

def _colour(code_id: int, max_id: int) -> str:
    """Map a code ID to an ANSI colour: blue=low, green=mid, red=high."""
    ratio = code_id / max(max_id - 1, 1)
    if ratio < 0.33:
        return "\033[94m"    # blue
    elif ratio < 0.67:
        return "\033[92m"    # green
    else:
        return "\033[91m"    # red

RESET = "\033[0m"


# ── Audio loading ─────────────────────────────────────────────────────────────

def load_audio(path: str, target_sr: int = 24_000) -> Tuple[np.ndarray, int, float]:
    """Load and resample audio. Returns (mono_float32, sr, duration_s)."""
    if not SOUNDFILE_AVAILABLE:
        raise RuntimeError("soundfile not installed. pip install soundfile")
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != target_sr:
        if not LIBROSA_AVAILABLE:
            raise RuntimeError(
                f"librosa needed to resample {sr}→{target_sr}. pip install librosa"
            )
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    duration = len(audio) / target_sr
    return audio, target_sr, duration


# ── Mel spectrogram features ──────────────────────────────────────────────────

def extract_mel_features(
    audio: np.ndarray,
    sr: int,
    n_frames: int,
    n_mels: int = 80,
    n_fft: int = 2048,
    hop_length: int = 240,
) -> np.ndarray:
    """
    Extract mel spectrogram and downsample to exactly n_frames.
    Returns (n_frames, n_mels) float32 array, values in [0, 1].
    """
    if LIBROSA_AVAILABLE:
        S = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, fmin=0.0, fmax=12_000.0,
        )
        S_db = librosa.power_to_db(S, ref=np.max)          # (n_mels, T_mel)
        S_db = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)
        # Resample mel frames → codec frames
        mel_t = S_db.shape[1]
        indices = np.linspace(0, mel_t - 1, n_frames).astype(int)
        features = S_db[:, indices].T                       # (n_frames, n_mels)
    else:
        # Fallback: simple energy per frame
        frame_len = len(audio) // n_frames
        features = np.zeros((n_frames, n_mels), dtype=np.float32)
        for i in range(n_frames):
            chunk = audio[i * frame_len: (i + 1) * frame_len]
            energy = float(np.mean(chunk ** 2)) if len(chunk) > 0 else 0.0
            features[i, :] = energy
        features = (features - features.min()) / (features.max() - features.min() + 1e-8)
    return features.astype(np.float32)


# ── Simulated realistic token assignment ──────────────────────────────────────

def simulate_trained_codes(
    features: np.ndarray,      # (T, n_mels)
    n_codebooks: int,
    codebook_size: int,
    seed: int = 42,
) -> np.ndarray:
    """
    Simulate what a well-trained model's token stream looks like.

    Strategy per codebook i:
      - Project mel features with a random (but fixed) matrix → codebook_dim
      - Apply softmax temperature to get a smooth probability over codes
      - Sample from that distribution so nearby frames share codes but
        transitions happen at acoustically meaningful boundaries

    This produces:
      - CB0: slow-changing codes (semantics / phonemes)
      - CB1: medium-changing codes (articulation)
      - CB2: faster-changing codes (prosody)
      - CB3: fastest-changing codes (fine detail)

    The result looks like a real trained model: structured, not uniform
    noise, but with genuine diversity across the codebook.
    """
    rng = np.random.default_rng(seed)
    T, n_mels = features.shape
    codes = np.zeros((T, n_codebooks), dtype=np.int32)

    # Temperature schedule: low temp (sharp) for CB0, higher for later CBs
    # This mirrors how residual VQ works: first CB captures most variance
    temps = [0.3, 0.6, 1.0, 1.5]
    temps = temps[:n_codebooks] + [1.5] * max(0, n_codebooks - len(temps))

    # Smoothing window: CB0 very smooth (phoneme-level), later CBs less so
    # Window in frames (10ms per frame): 200ms, 80ms, 40ms, 20ms
    smooth_frames = [20, 8, 4, 2]
    smooth_frames = smooth_frames[:n_codebooks] + [2] * max(0, n_codebooks - len(smooth_frames))

    # Random projection matrices (fixed per codebook by seed)
    proj_dim = min(n_mels, codebook_size)
    residual = features.copy()

    for cb in range(n_codebooks):
        proj = rng.standard_normal((n_mels, proj_dim)).astype(np.float32)
        proj /= np.linalg.norm(proj, axis=0, keepdims=True) + 1e-8

        # Project residual features
        projected = residual @ proj                        # (T, proj_dim)

        # Smooth temporally (simulates the causal encoder's temporal context)
        w = smooth_frames[cb]
        smoothed = np.zeros_like(projected)
        for t in range(T):
            lo = max(0, t - w)
            smoothed[t] = projected[lo:t+1].mean(axis=0)

        # Map smoothed features to codebook logits via random codebook matrix
        cb_matrix = rng.standard_normal((proj_dim, codebook_size)).astype(np.float32)
        cb_matrix /= np.linalg.norm(cb_matrix, axis=0, keepdims=True) + 1e-8
        logits = smoothed @ cb_matrix                      # (T, codebook_size)

        # Temperature softmax → sample codes
        temp = temps[cb]
        logits_t = logits / temp
        logits_t -= logits_t.max(axis=1, keepdims=True)   # numerical stability
        probs = np.exp(logits_t)
        probs /= probs.sum(axis=1, keepdims=True)

        frame_codes = np.array([
            rng.choice(codebook_size, p=probs[t]) for t in range(T)
        ], dtype=np.int32)
        codes[:, cb] = frame_codes

        # Update residual: subtract the "quantised" contribution
        # (mimics how RVQ removes the coarse component at each stage)
        assigned_vecs = cb_matrix[:, frame_codes].T        # (T, proj_dim)
        residual = residual - (assigned_vecs @ proj.T) * 0.5

    return codes


# ── Real model encode ─────────────────────────────────────────────────────────

def encode_with_real_model(
    checkpoint_path: str,
    audio_path: str,
    lang: str = "hi",
) -> Tuple[np.ndarray, int, int, float]:
    """
    Load a trained Lipika checkpoint and encode the audio.
    Returns (codes_np, n_codebooks, codebook_size, frame_rate).
    codes_np shape: (T, n_codebooks)
    """
    # Import from the main module — expected to be in the same directory
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from lipika_tokenizer_fixed import (
            _load_model_from_checkpoint, encode_audio_file,
        )
    except ImportError:
        raise RuntimeError(
            "lipika_tokenizer_fixed.py not found in the same directory. "
            "Place lipika_viewer.py next to lipika_tokenizer_fixed.py."
        )

    print("  Loading checkpoint …")
    model = _load_model_from_checkpoint(checkpoint_path)
    n_cb  = model.rvq_cfg.n_codebooks
    cb_sz = model.rvq_cfg.codebook_size
    fr    = model.frame_rate

    print("  Encoding audio …")
    codes_t = encode_audio_file(model, audio_path, lang=lang, diagnostic=False)
    codes_np = codes_t[0].cpu().numpy()    # (T, n_cb)
    return codes_np, n_cb, cb_sz, fr


# ── Display ───────────────────────────────────────────────────────────────────

CB_LABELS = [
    "Semantic/Phoneme",
    "Articulation",
    "Prosody/Tone",
    "Timbre/Voice",
]


def display(
    codes: np.ndarray,            # (T, n_cb)
    n_frames_show: int,
    codebook_size: int,
    frame_rate: float,
    audio_path: str,
    duration: float,
    n_samples: int,
    mode_label: str,
) -> None:
    T, n_cb = codes.shape
    show = min(n_frames_show, T)
    ms_per_frame = 1000.0 / frame_rate

    print()
    width = show * 4 + 40
    print("=" * width)
    print(f"  AUDIO TOKEN MAP — {T} frames × {n_cb} codebooks  [{mode_label}]")
    print("=" * width)

    # Time header
    time_header = "Time →  " + "".join(
        f"{i:<4}" if i % 10 == 0 else "    " for i in range(show)
    )
    ms_header = "ms   →  " + "".join(
        f"{int(i*ms_per_frame):<4}" if i % 10 == 0 else "    " for i in range(show)
    )
    print(time_header)
    print(ms_header)
    print("    " + "─" * (show * 4 + 36))

    for cb in range(n_cb):
        label = CB_LABELS[cb] if cb < len(CB_LABELS) else f"CB{cb}"
        row_parts = []
        for t in range(show):
            cid = int(codes[t, cb])
            col = _colour(cid, codebook_size)
            row_parts.append(f"{col}{cid:3d}{RESET}")
        row = f"CB{cb} ({label:20s}) │ " + "  ".join(row_parts)
        print(row)

    print("    " + "─" * (show * 4 + 36))
    print()
    print(f"  Legend: {_colour(0, codebook_size)}blue=low{RESET}  "
          f"{_colour(codebook_size//2, codebook_size)}green=mid{RESET}  "
          f"{_colour(codebook_size-1, codebook_size)}red=high{RESET}")

    # Stats
    print()
    print("=" * 72)
    print("  CODEBOOK STATISTICS")
    print("=" * 72)
    print(f"  {'CB':<5} {'Unique':>8} {'Usage %':>9} {'Entropy':>9} "
          f"{'Perplexity':>12} {'Unused':>8}  Status")
    print("  " + "-" * 68)

    for cb in range(n_cb):
        flat   = codes[:, cb].astype(int)
        counts = np.bincount(flat, minlength=codebook_size)
        unique = int((counts > 0).sum())
        usage  = unique / codebook_size * 100
        probs  = counts / counts.sum()
        pnz    = probs[probs > 0]
        ent    = float(-np.sum(pnz * np.log(pnz + 1e-12)))
        perp   = float(np.exp(ent))
        unused = codebook_size - unique
        ok     = unique >= max(2, codebook_size // 4)
        status = "✅ OK" if ok else "⚠️  LOW"
        print(f"  {cb:<5} {unique:>8} {usage:>8.1f}%  {ent:>9.3f}  {perp:>12.1f}  "
              f"{unused:>8}  {status}")

    # Token sequence table
    print()
    print("=" * 72)
    print(f"  CODE SEQUENCE — Frames 0 to {min(29, T-1)}")
    print("=" * 72)
    print(f"  {'Frame':>6}  {'ms':>6} | " + "  ".join(f"CB{i}" for i in range(n_cb))
          + " | Combined Token")
    print("  " + "-" * 60)

    for t in range(min(30, T)):
        ms     = int(t * ms_per_frame)
        cids   = [int(codes[t, cb]) for cb in range(n_cb)]
        combined = sum(cids[cb] * (codebook_size ** (n_cb - 1 - cb)) for cb in range(n_cb))
        cid_str = "  ".join(f"{c:3d}" for c in cids)
        print(f"  {t:>6}  {ms:>5}ms | {cid_str} | {combined:>14d}")

    print("  " + "-" * 60)

    unique_combined = len(set(
        sum(int(codes[t, cb]) * (codebook_size ** (n_cb - 1 - cb)) for cb in range(n_cb))
        for t in range(T)
    ))
    print(f"\n  Unique combined tokens: {unique_combined}/{T} "
          f"({unique_combined/T*100:.1f}%)")
    print(f"  Max possible:           {codebook_size**n_cb:,d} "
          f"({codebook_size}^{n_cb})")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lipika audio token viewer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--audio",            required=True,  help="Path to audio file")
    parser.add_argument("--checkpoint",       default=None,   help="Trained checkpoint (.pt). Omit for diagnostic mode.")
    parser.add_argument("--lang",             default="hi",   help="ISO 639-1 language code")
    parser.add_argument("--n-frames",         type=int, default=60,   help="Frames to display in the token map")
    parser.add_argument("--codebook-size",    type=int, default=32,   help="(diagnostic mode) codebook size")
    parser.add_argument("--n-codebooks",      type=int, default=4,    help="(diagnostic mode) number of codebooks")
    parser.add_argument("--frame-rate",       type=float, default=100.0, help="(diagnostic mode) frames per second")
    parser.add_argument("--sample-rate",      type=int, default=24_000, help="Target sample rate")
    parser.add_argument("--seed",             type=int, default=42)
    args = parser.parse_args()

    print()
    print("=" * 72)
    if args.checkpoint:
        print("  LIPIKA CODE VIEWER — Trained Model")
        mode_label = "TRAINED MODEL"
    else:
        print("  LIPIKA CODE VIEWER — Diagnostic Mode (simulated tokens)")
        mode_label = "DIAGNOSTIC (audio-correlated simulation)"
    print("=" * 72)

    # Load audio
    print(f"\n  Loading: {args.audio}")
    try:
        audio, sr, duration = load_audio(args.audio, target_sr=args.sample_rate)
    except Exception as e:
        print(f"  ERROR loading audio: {e}")
        sys.exit(1)

    print(f"  Duration : {duration:.2f}s")
    print(f"  Samples  : {len(audio):,d}")

    if args.checkpoint:
        # ── Real model path ──
        try:
            codes, n_cb, cb_sz, frame_rate = encode_with_real_model(
                args.checkpoint, args.audio, lang=args.lang
            )
        except Exception as e:
            print(f"\n  ERROR encoding with checkpoint: {e}")
            sys.exit(1)

        T = codes.shape[0]
        print(f"  Frames   : {T}  ({frame_rate:.1f} Hz)")
        print(f"  Codebooks: {n_cb} × {cb_sz}")
        display(codes, args.n_frames, cb_sz, frame_rate,
                args.audio, duration, len(audio), mode_label)

        print()
        print("=" * 72)
        print("  These are REAL tokens from a trained model.")
        print("=" * 72)

    else:
        # ── Diagnostic path ──
        n_cb  = args.n_codebooks
        cb_sz = args.codebook_size
        fr    = args.frame_rate
        T     = int(duration * fr)

        print(f"  Frames   : {T}  ({fr:.0f} Hz codec, {args.sample_rate//int(fr)} samples/frame)")
        print(f"  Codebooks: {n_cb} × {cb_sz}")
        print()
        print("  Extracting audio features for simulation …")
        features = extract_mel_features(audio, sr, T)

        print("  Simulating trained-model token assignments …")
        codes = simulate_trained_codes(features, n_cb, cb_sz, seed=args.seed)

        display(codes, args.n_frames, cb_sz, fr,
                args.audio, duration, len(audio), mode_label)

        print()
        print("=" * 72)
        print("  DIAGNOSTIC MODE — tokens are simulated, not from a real model.")
        print("  The pattern IS audio-correlated (driven by mel spectrogram).")
        print("  CB0 changes slowly (phoneme-rate), CB3 fastest (frame-rate).")
        print("  Train the model to get real tokens: python lipika_tokenizer_fixed.py train")
        print("=" * 72)


if __name__ == "__main__":
    main()