#!/usr/bin/env python3
# =============================================================================
# LIPIKA TOKENIZER  —  Step 1: AudioPreprocessor
# =============================================================================
#
# Purpose
# -------
# Converts ANY audio file (wav / flac / ogg / mp3 / opus) into a normalised
# 24 kHz mono waveform tensor with shape (1, 1, T) ready to feed directly
# into LipikaTokenizer's AudioEncoder.
#
# This is intentionally the ONLY responsibility of this class:
#   1. Load raw audio bytes from disk
#   2. Downmix to mono  (average channels)
#   3. Resample to 24 000 Hz  (if the file is at a different rate)
#   4. Peak-normalise to ±0.98 (same convention as AudioDataset)
#   5. Optionally trim / pad to a fixed duration
#   6. Return a float32 tensor  (B=1, C=1, T)
#
# Everything else (quantisation, encoding, script conditioning) lives
# in the downstream modules.
#
# Design notes
# ------------
# * Depends only on soundfile + librosa — both are already required by the
#   rest of the project.
# * Stateless: call process() as many times as you like; no internal state
#   accumulates.
# * All errors surface as typed exceptions (AudioLoadError, ResampleError)
#   so callers can handle them precisely instead of catching bare Exception.
#
# Usage
# -----
#   prep = AudioPreprocessor(target_sr=24_000)
#
#   # From a file path
#   waveform = prep.process("speech_hindi.wav")        # → (1, 1, T) tensor
#
#   # With a fixed duration (will trim or pad)
#   waveform = prep.process("speech_hindi.wav", max_duration=5.0)
#
#   # From a numpy array you already have in memory
#   waveform = prep.from_numpy(array, original_sr=16_000)
#
#   # Batch of files
#   batch = prep.process_batch(["a.wav", "b.flac"], max_duration=5.0)  # (N, 1, T)
#
# =============================================================================

from __future__ import annotations

import warnings
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

# ── Hard dependencies ────────────────────────────────────────────────────────
try:
    import soundfile as sf
except ImportError as e:
    raise ImportError(
        "soundfile is required by AudioPreprocessor.\n"
        "Install it with:  pip install soundfile"
    ) from e

try:
    import librosa
except ImportError as e:
    raise ImportError(
        "librosa is required by AudioPreprocessor.\n"
        "Install it with:  pip install librosa"
    ) from e


# =============================================================================
# TYPED EXCEPTIONS
# =============================================================================

class AudioLoadError(RuntimeError):
    """Raised when a file cannot be opened or decoded."""


class ResampleError(RuntimeError):
    """Raised when resampling fails (unsupported rate, corrupt audio, …)."""


# =============================================================================
# AUDIO PREPROCESSOR
# =============================================================================

class AudioPreprocessor:
    """
    Single-responsibility class: any audio file → 24 kHz mono waveform tensor.

    Parameters
    ----------
    target_sr : int
        Output sample rate.  Default 24 000 Hz matches the rest of Lipika.
    peak_norm : bool
        If True, peak-normalise each clip to ±0.98 (same as AudioDataset).
        Set False if you need raw amplitude (e.g. for dB-level analysis).
    resample_quality : str
        librosa resampling algorithm passed as ``res_type``.
        'soxr_hq'  → high quality, slightly slower  (default)
        'kaiser_best' → highest quality, slowest
        'soxr_mq'  → medium quality, fast
        'linear'   → very fast, lower quality (good for smoke-tests)

    Attributes
    ----------
    target_sr : int  (read-only)
    SUPPORTED_EXTENSIONS : frozenset[str]
    """

    SUPPORTED_EXTENSIONS: frozenset = frozenset(
        {".wav", ".flac", ".ogg", ".mp3", ".opus", ".aiff", ".aif", ".au"}
    )

    def __init__(
        self,
        target_sr: int = 24_000,
        peak_norm: bool = True,
        resample_quality: str = "soxr_hq",
    ) -> None:
        if target_sr <= 0:
            raise ValueError(f"target_sr must be a positive integer, got {target_sr}")
        self._target_sr = target_sr
        self._peak_norm = peak_norm
        self._resample_quality = resample_quality

    # ── Public read-only property ─────────────────────────────────────────

    @property
    def target_sr(self) -> int:
        return self._target_sr

    # ── Primary API ──────────────────────────────────────────────────────

    def process(
        self,
        audio_path: Union[str, Path],
        max_duration: Optional[float] = None,
        start_offset: float = 0.0,
    ) -> torch.Tensor:
        """
        Load, resample and normalise a single audio file.

        Parameters
        ----------
        audio_path : str | Path
            Path to the audio file.
        max_duration : float | None
            If given, the output is trimmed or zero-padded to exactly
            ``max_duration`` seconds after resampling.
            ``None`` → return the full file (no truncation / padding).
        start_offset : float
            Start reading from this many seconds into the file.
            Ignored if max_duration is None.

        Returns
        -------
        torch.Tensor  –  shape (1, 1, T)  dtype float32
            Batch dimension (1), channel dimension (1), time samples (T).
            T = target_sr * max_duration   when max_duration is not None,
            T = len(audio_file) * target_sr / original_sr  otherwise.

        Raises
        ------
        FileNotFoundError
            If the path does not exist.
        ValueError
            If the file extension is not in SUPPORTED_EXTENSIONS.
        AudioLoadError
            If soundfile cannot decode the file.
        ResampleError
            If librosa resampling fails.
        """
        path = Path(audio_path)
        self._validate_path(path)

        # 1. Load from disk → float32 numpy array
        raw, original_sr = self._load_raw(path)

        # 2. Downmix to mono
        mono = self._to_mono(raw)

        # 3. Apply start offset (in samples of the *original* rate)
        if start_offset > 0.0 and max_duration is not None:
            offset_samples = int(start_offset * original_sr)
            mono = mono[offset_samples:]

        # 4. Resample to target_sr
        audio = self._resample(mono, original_sr)

        # 5. Trim / pad to fixed duration (optional)
        if max_duration is not None:
            audio = self._fix_length(audio, max_duration)

        # 6. Peak normalise
        if self._peak_norm:
            audio = self._normalise(audio)

        # 7. Wrap in (1, 1, T) tensor
        return torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0)

    def from_numpy(
        self,
        array: np.ndarray,
        original_sr: int,
        max_duration: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Same pipeline as process() but starting from a numpy array you
        already have in memory (e.g. from a streaming source or a test stub).

        Parameters
        ----------
        array : np.ndarray
            Raw audio.  Shape: (T,) mono  OR  (T, C) / (C, T) multi-channel.
            Will be downmixed to mono automatically.
        original_sr : int
            The sample rate of ``array``.
        max_duration : float | None
            Optional fixed-length output in seconds.

        Returns
        -------
        torch.Tensor  –  shape (1, 1, T)  dtype float32
        """
        array = np.asarray(array, dtype=np.float32)

        # Handle (C, T) layout from some libraries
        if array.ndim == 2 and array.shape[0] < array.shape[1]:
            array = array.T          # → (T, C)

        mono  = self._to_mono(array)
        audio = self._resample(mono, original_sr)

        if max_duration is not None:
            audio = self._fix_length(audio, max_duration)
        if self._peak_norm:
            audio = self._normalise(audio)

        return torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0)

    def process_batch(
        self,
        audio_paths: List[Union[str, Path]],
        max_duration: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Convenience wrapper: process a list of files and stack into one batch.

        All clips are processed with the same ``max_duration`` so they can
        be stacked. If ``max_duration`` is None, all clips must have the
        same number of samples after resampling.

        Parameters
        ----------
        audio_paths : list[str | Path]
        max_duration : float | None

        Returns
        -------
        torch.Tensor  –  shape (N, 1, T)  dtype float32
        """
        tensors = [self.process(p, max_duration=max_duration) for p in audio_paths]
        return torch.cat(tensors, dim=0)   # (N, 1, T)

    # ── Private helpers ───────────────────────────────────────────────────

    def _validate_path(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported audio format '{path.suffix}'. "
                f"Supported: {sorted(self.SUPPORTED_EXTENSIONS)}"
            )

    def _load_raw(self, path: Path) -> tuple[np.ndarray, int]:
        """
        Read audio file to float32 numpy array using soundfile.
        Falls back to librosa.load for formats soundfile cannot handle
        (e.g. MP3, some OGG variants).

        Returns (array_shape_T_C_or_T, original_sample_rate).
        """
        try:
            audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
            # audio shape: (T, C)
            return audio, sr
        except Exception as sf_err:
            # soundfile cannot decode this format (MP3, some codecs) — try librosa
            try:
                audio, sr = librosa.load(str(path), sr=None, mono=False, dtype=np.float32)
                # librosa returns (C, T) or (T,) — normalise to (T, C)
                if audio.ndim == 1:
                    audio = audio[:, np.newaxis]
                else:
                    audio = audio.T
                return audio, sr
            except Exception as lr_err:
                raise AudioLoadError(
                    f"Cannot load '{path}'.\n"
                    f"soundfile error : {sf_err}\n"
                    f"librosa  error  : {lr_err}"
                ) from lr_err

    @staticmethod
    def _to_mono(audio: np.ndarray) -> np.ndarray:
        """
        Downmix to mono by averaging channels.

        Accepts both (T,) and (T, C) shapes; always returns (T,).
        """
        if audio.ndim == 1:
            return audio
        if audio.shape[1] == 1:
            return audio[:, 0]
        # Average across channels — standard downmix, preserves loudness
        return audio.mean(axis=1)

    def _resample(self, mono: np.ndarray, original_sr: int) -> np.ndarray:
        """
        Resample mono audio array from original_sr to self.target_sr.
        No-op if original_sr == target_sr (avoids unnecessary compute).
        """
        if original_sr == self._target_sr:
            return mono
        try:
            return librosa.resample(
                mono,
                orig_sr=original_sr,
                target_sr=self._target_sr,
                res_type=self._resample_quality,
            )
        except Exception as e:
            raise ResampleError(
                f"Resampling from {original_sr} Hz to {self._target_sr} Hz failed: {e}"
            ) from e

    def _fix_length(self, audio: np.ndarray, max_duration: float) -> np.ndarray:
        """
        Trim or zero-pad ``audio`` to exactly max_duration seconds.
        Trimming takes the first max_samples; padding appends zeros.
        """
        max_samples = int(max_duration * self._target_sr)
        if len(audio) >= max_samples:
            return audio[:max_samples]
        pad_width = max_samples - len(audio)
        return np.pad(audio, (0, pad_width), mode="constant", constant_values=0.0)

    @staticmethod
    def _normalise(audio: np.ndarray) -> np.ndarray:
        """
        Peak-normalise to ±0.98.

        Clips with near-silence (peak < 1e-6) are left untouched to avoid
        amplifying noise to full scale.
        """
        peak = np.abs(audio).max()
        if peak > 1e-6:
            audio = audio / (peak + 1e-6) * 0.98
        return audio

    # ── Diagnostics ───────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"AudioPreprocessor("
            f"target_sr={self._target_sr}, "
            f"peak_norm={self._peak_norm}, "
            f"resample_quality='{self._resample_quality}')"
        )

    def inspect(self, audio_path: Union[str, Path]) -> dict:
        """
        Return metadata about a file WITHOUT loading the full waveform.
        Useful for debugging or dataset validation.

        Returns
        -------
        dict with keys: path, original_sr, channels, duration_s,
                        original_samples, output_samples, needs_resample
        """
        path = Path(audio_path)
        self._validate_path(path)
        info = sf.info(str(path))
        return {
            "path":             str(path),
            "original_sr":      info.samplerate,
            "channels":         info.channels,
            "duration_s":       info.duration,
            "original_samples": info.frames,
            "output_samples":   int(info.duration * self._target_sr),
            "needs_resample":   info.samplerate != self._target_sr,
            "format":           info.format,
            "subtype":          info.subtype,
        }


# =============================================================================
# QUICK SELF-TEST  (run with: python audio_preprocessor.py)
# =============================================================================

if __name__ == "__main__":
    import tempfile
    import os

    print("=" * 60)
    print("  AudioPreprocessor — self-test")
    print("=" * 60)

    # ── 1. Create a synthetic WAV at 16 kHz (simulates a common ASR corpus rate)
    sr_orig = 16_000
    duration = 3.0
    t = np.linspace(0, duration, int(sr_orig * duration), dtype=np.float32)
    # Simulate a voiced speech-like signal: F0=120 Hz + harmonics
    signal = (
        0.4 * np.sin(2 * np.pi * 120 * t)
        + 0.2 * np.sin(2 * np.pi * 240 * t)
        + 0.1 * np.sin(2 * np.pi * 360 * t)
    )
    # Make it stereo to test downmix
    stereo = np.stack([signal, signal * 0.8], axis=1)   # (T, 2)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    sf.write(tmp_path, stereo, sr_orig)
    print(f"\n[1] Wrote synthetic 16kHz stereo WAV: {tmp_path}")
    print(f"    Shape: {stereo.shape}  SR: {sr_orig} Hz  Duration: {duration}s")

    # ── 2. Instantiate preprocessor
    prep = AudioPreprocessor(target_sr=24_000, peak_norm=True)
    print(f"\n[2] {prep}")

    # ── 3. inspect() — no load, just metadata
    meta = prep.inspect(tmp_path)
    print(f"\n[3] inspect():")
    for k, v in meta.items():
        print(f"    {k:<20}: {v}")

    # ── 4. process() — full pipeline
    waveform = prep.process(tmp_path)
    expected_T = int(sr_orig * duration * 24_000 / sr_orig)   # = 72000
    print(f"\n[4] process() (no max_duration):")
    print(f"    Output shape  : {tuple(waveform.shape)}   (expected (1, 1, ~{expected_T}))")
    print(f"    dtype         : {waveform.dtype}")
    print(f"    Peak value    : {waveform.abs().max().item():.4f}  (should be ≈0.98)")

    # ── 5. process() with max_duration
    waveform_2s = prep.process(tmp_path, max_duration=2.0)
    expected_2s = 24_000 * 2
    print(f"\n[5] process(max_duration=2.0s):")
    print(f"    Output shape  : {tuple(waveform_2s.shape)}   (expected (1, 1, {expected_2s}))")

    # ── 6. from_numpy()
    waveform_np = prep.from_numpy(signal, original_sr=sr_orig, max_duration=1.0)
    print(f"\n[6] from_numpy(1-channel, 16kHz, max_duration=1.0s):")
    print(f"    Output shape  : {tuple(waveform_np.shape)}   (expected (1, 1, {24_000}))")

    # ── 7. process_batch()
    batch = prep.process_batch([tmp_path, tmp_path], max_duration=2.0)
    print(f"\n[7] process_batch([file, file], max_duration=2.0s):")
    print(f"    Output shape  : {tuple(batch.shape)}   (expected (2, 1, {expected_2s}))")

    # ── 8. Error handling
    print(f"\n[8] Error handling:")
    try:
        prep.process("nonexistent.wav")
    except FileNotFoundError as e:
        print(f"    FileNotFoundError ✓  ({e})")

    try:
        prep.process(tmp_path.replace(".wav", ".xyz"))
    except (FileNotFoundError, ValueError) as e:
        print(f"    ValueError (bad ext) ✓  ({type(e).__name__})")

    # ── Cleanup
    os.unlink(tmp_path)
    print(f"\n[OK] All tests passed — AudioPreprocessor is ready.\n")
    print("Next step: feed waveform into AudioEncoder.")
    print("  waveform shape (1, 1, T) feeds directly into AudioEncoder.forward(waveform)")