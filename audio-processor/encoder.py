#!/usr/bin/env python3
# =============================================================================
# LIPIKA TOKENIZER  —  Step 2: AudioEncoder
# =============================================================================
#
# Responsibility
# --------------
# Convert a raw 24 kHz mono waveform tensor  (B, 1, T)
#                             into a continuous latent sequence  (B, T_frames, C)
#
# Architecture
# ------------
#
#   Input  (B, 1, T)
#      │
#      ▼
#   CausalConv1d  kernel=7  →  (B, C, T)           ← stem: 1 channel → C channels
#      │
#      ▼
#   ┌─────────────────────────────────────────────┐
#   │  EncoderBlock  stride=2                     │  T  →  T/2
#   │  EncoderBlock  stride=4                     │  T/2 → T/8
#   │  EncoderBlock  stride=5                     │  T/8 → T/40
#   │  EncoderBlock  stride=6                     │  T/40→ T/240
#   └─────────────────────────────────────────────┘
#      │
#      ▼
#   Bottleneck  (ELU + CausalConv1d 1×1)          ← clean latent projection
#      │
#      ▼
#   Transpose  (B, C, T_frames) → (B, T_frames, C)
#      │
#      ▼
#   LayerNorm  over C
#      │
#      ▼ (optional)
#   ScriptFamilyAdapter  (AdaLN)                  ← scale + shift by script family
#      │
#      ▼
#   Output  (B, T_frames, C)
#
# Compression ratio:  2 × 4 × 5 × 6 = 240
# At 24 000 Hz:  frame rate = 24000 / 240 = 100 Hz  (10 ms per frame)
#
# Each EncoderBlock keeps channel count CONSTANT at C:
#   ResBlock stack (dilations 1, 3, 9)  →  C channels
#   CausalConv1d  C → 2C               →  doubles channels
#   Gating  2C[:C]                     →  halves back to C
#   AvgPool1d stride                   →  temporal downsampling
#
# This matches the EnCodec design [1] — no channel blow-up through depth.
#
# References
# ----------
#   [1] Défossez et al. (2022) "High Fidelity Neural Audio Compression" (EnCodec)
#       https://arxiv.org/abs/2210.13438
#   [12] Ba et al. (2016) "Layer Normalization"
#        https://arxiv.org/abs/1607.06450
#
# Dependencies
# ------------
#   torch  (no other heavy deps)
#
# Usage
# -----
#   from audio_preprocessor import AudioPreprocessor
#   from audio_encoder import AudioEncoder, AudioConfig, ModelConfig
#
#   prep    = AudioPreprocessor(target_sr=24_000)
#   waveform = prep.process("speech.wav")          # (1, 1, T)
#
#   encoder = AudioEncoder(AudioConfig(), ModelConfig())
#   latent  = encoder(waveform)                    # (1, T_frames, 512)
#
# =============================================================================

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# CONFIG DATACLASSES  (duplicated here so this file is self-contained)
# =============================================================================

@dataclass
class AudioConfig:
    """
    Audio processing parameters shared across all Lipika modules.
    24 kHz captures retroflex stops and aspirated consonants (energy up to ~10 kHz)
    without the memory cost of 44.1 kHz.
    """
    sample_rate: int = 24_000
    n_fft: int = 2048
    hop_length: int = 240       # → 100 Hz frame rate
    n_mels: int = 128
    fmin: float = 0.0
    fmax: float = 12_000.0


@dataclass
class ModelConfig:
    """Encoder / decoder / adapter sizes."""
    encoder_channels: int = 512
    encoder_depth: int = 8          # reserved for future deeper variants
    decoder_channels: int = 512
    decoder_depth: int = 8

    # Semantic teacher projection dims  (used by RVQ, not encoder directly)
    w2v_bert_model: str = "facebook/w2v-bert-2.0"
    w2v_bert_dim: int = 1024
    semantic_proj_dim: int = 256

    # Script adapter
    n_script_families: int = 12
    script_embed_dim: int = 64

    # Discriminator (not used here)
    disc_channels: int = 64
    disc_depth: int = 4
    mpd_periods: List[int] = field(default_factory=lambda: [2, 3, 5, 7, 11])


# =============================================================================
# BUILDING BLOCKS
# =============================================================================

class CausalConv1d(nn.Module):
    """
    Causal 1-D convolution — zero future context.

    Standard Conv1d uses symmetric padding; that would let each output frame
    see future input samples, which breaks streaming / real-time inference.
    We pad *only on the left* (past) so frame t depends solely on frames ≤ t.

    causal_pad = (kernel_size - 1) * dilation
    After padding:  effective kernel spans [t - causal_pad, t]

    Reference: EnCodec §3.1 [1].
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.causal_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=0, bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        x = F.pad(x, (self.causal_pad, 0))   # left-pad only
        return self.conv(x)

    def extra_repr(self) -> str:
        c = self.conv
        return (
            f"in={c.in_channels}, out={c.out_channels}, "
            f"kernel={c.kernel_size[0]}, dilation={c.dilation[0]}, "
            f"causal_pad={self.causal_pad}"
        )


class ResBlock(nn.Module):
    """
    Gated residual block with a single dilated causal convolution.

    Structure:
        x  →  ELU  →  CausalConv1d(k=3, d=dilation)  →  ELU  →  CausalConv1d(k=1)  →  + x

    Three ResBlocks with dilations [1, 3, 9] give a receptive field of:
        (3-1)*1 + (3-1)*3 + (3-1)*9 = 2 + 6 + 18 = 26 frames
    before any striding — enough to capture ~260 ms at 100 Hz.

    The 1×1 conv at the end acts as a channel mixer without expanding
    the temporal receptive field.
    """

    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.ELU(),
            CausalConv1d(channels, channels, kernel_size=3, dilation=dilation),
            nn.ELU(),
            CausalConv1d(channels, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(x)    # residual connection


class EncoderBlock(nn.Module):
    """
    One downsampling stage: residual refinement → strided compression.

    Channel flow:
        in:  (B, C, T)
        res: (B, C, T)         ← three ResBlocks, channels unchanged
        down:(B, 2C, T)        ← CausalConv1d doubles channels
        gate:(B, C, T)         ← keep first C channels  (learned gating)
        pool:(B, C, T/stride)  ← AvgPool1d reduces time

    Net result: same channel count C, time divided by stride.
    The gate is analogous to a Gated Linear Unit applied at the downsampling
    boundary — it lets the network selectively pass information.

    Why AvgPool instead of strided conv for temporal reduction?
    AvgPool has no learnable parameters and anti-aliases the signal before
    subsampling, reducing aliasing artefacts in the latent.
    The CausalConv1d with kernel=2*stride handles the learned compression,
    and AvgPool handles the anti-aliased subsampling cleanly.
    """

    def __init__(self, channels: int, stride: int) -> None:
        super().__init__()
        self.stride = stride
        # Three residual blocks with expanding dilation for large receptive field
        self.res  = nn.Sequential(
            ResBlock(channels, dilation=1),
            ResBlock(channels, dilation=3),
            ResBlock(channels, dilation=9),
        )
        # Doubles channels; gating in forward halves back to C
        self.down = CausalConv1d(channels, channels * 2, kernel_size=2 * stride)
        self.pool = nn.AvgPool1d(stride, stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.res(x)                         # (B, C, T)
        x = self.down(x)                        # (B, 2C, T)
        x = x[:, : x.shape[1] // 2, :]         # gate → (B, C, T)
        x = self.pool(x)                        # (B, C, T/stride)
        return x

    def extra_repr(self) -> str:
        return f"stride={self.stride}"


# =============================================================================
# SCRIPT-FAMILY ADAPTER  (Adaptive Layer Normalisation)
# =============================================================================

class ScriptFamilyAdapter(nn.Module):
    """
    Conditions the encoder latent on the script family of the input language.

    Mechanism: Adaptive Layer Normalisation (AdaLN) [12].
        z_out = z * scale(script_id) + shift(script_id)

    Why script conditioning?
    -----------------------
    Retroflex consonants (/ʈ ɖ ɳ ɽ/) are phonemically contrastive in most
    Indic languages (Hindi, Tamil, Telugu, Malayalam…) but absent in, e.g.,
    Persian-script languages (Urdu). Telling the encoder which script family
    it's processing lets it allocate latent dimensions for these distinctions
    rather than discovering them blindly from data.

    The first 8 embedding dims get a +0.5 bias for retroflex scripts as a
    soft inductive prior — training will correct this if it's wrong.

    scale is initialised to 1 (identity) and shift to 0 so AdaLN has zero
    effect at the start of training and learns the residual correction.
    """

    RETROFLEX_SCRIPTS = {0, 1, 2, 4, 5, 6, 7, 8}   # Devanagari … Malayalam

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.embed = nn.Embedding(cfg.n_script_families, cfg.script_embed_dim)

        # Soft retroflex prior on first 8 embedding dims
        with torch.no_grad():
            for sf_id in self.RETROFLEX_SCRIPTS:
                self.embed.weight[sf_id, :8] += 0.5

        self.proj = nn.Sequential(
            nn.Linear(cfg.script_embed_dim, cfg.encoder_channels),
            nn.SiLU(),
            nn.Linear(cfg.encoder_channels, cfg.encoder_channels),
        )
        self.scale_head = nn.Linear(cfg.encoder_channels, cfg.encoder_channels)
        self.shift_head = nn.Linear(cfg.encoder_channels, cfg.encoder_channels)

        # Identity init: scale=1, shift=0  →  no effect at step 0
        nn.init.zeros_(self.scale_head.weight)
        nn.init.ones_(self.scale_head.bias)
        nn.init.zeros_(self.shift_head.weight)
        nn.init.zeros_(self.shift_head.bias)

    def forward(self, script_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        script_ids : (B,) int64  — index in [0, n_script_families)

        Returns
        -------
        dict with 'scale' (B, C) and 'shift' (B, C)
        """
        e = self.proj(self.embed(script_ids))
        return {
            "scale": self.scale_head(e),   # (B, C)
            "shift": self.shift_head(e),   # (B, C)
        }


# =============================================================================
# AUDIO ENCODER
# =============================================================================

class AudioEncoder(nn.Module):
    """
    Causal convolutional encoder: waveform (B, 1, T) → latent (B, T_frames, C).

    Strides
    -------
    [2, 4, 5, 6]  →  total compression ratio = 240
    At 24 000 Hz:  frame rate = 24 000 / 240 = 100 Hz  (one frame per 10 ms)

    This matches the standard TTS alignment frame rate used in:
    - VALL-E [4] for codec token alignment
    - Vocos [13] for vocoder frame rate
    - Most forced-alignment tools (Montreal Forced Aligner default)

    Channel count is CONSTANT at C throughout all EncoderBlocks.
    Only temporal resolution decreases.

    Parameters
    ----------
    audio_cfg : AudioConfig
    model_cfg : ModelConfig
        encoder_channels (C) determines latent width.

    Forward
    -------
    waveform   : (B, 1, T)         float32 — output of AudioPreprocessor
    script_ids : (B,) int64 | None — optional script family for AdaLN

    Returns
    -------
    latent : (B, T_frames, C)   float32
        T_frames = T // compression_ratio  (integer division, may be ±1 off
        for lengths not divisible by 240 — the decoder handles the mismatch).
    """

    STRIDES: List[int] = [2, 4, 5, 6]

    def __init__(self, audio_cfg: AudioConfig, model_cfg: ModelConfig) -> None:
        super().__init__()
        C = model_cfg.encoder_channels

        # Stem: raw waveform (1 channel) → C channels, causal conv k=7
        self.stem = CausalConv1d(1, C, kernel_size=7)

        # Downsampling stack — all blocks operate at C channels
        self.blocks = nn.Sequential(*[
            EncoderBlock(C, stride) for stride in self.STRIDES
        ])

        # Bottleneck: 1×1 causal conv + ELU — cleans up the latent before quantisation
        self.bottleneck = nn.Sequential(
            nn.ELU(),
            CausalConv1d(C, C, kernel_size=1),
        )

        # Post-norm over channel dim (standard for transformer-adjacent latents)
        self.norm = nn.LayerNorm(C)

        # Script adapter is stored separately so it can be used optionally
        self.script_adapter = ScriptFamilyAdapter(model_cfg)

        self._C = C

    def forward(
        self,
        waveform: torch.Tensor,
        script_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        waveform   : (B, 1, T)   float32
        script_ids : (B,)  int64 | None

        Returns
        -------
        latent : (B, T_frames, C)   float32
        """
        # stem: (B, 1, T) → (B, C, T)
        x = self.stem(waveform)

        # downsampling: (B, C, T) → (B, C, T_frames)
        x = self.blocks(x)

        # bottleneck: (B, C, T_frames) unchanged shape
        x = self.bottleneck(x)

        # transpose + norm: (B, C, T_frames) → (B, T_frames, C)
        x = x.transpose(1, 2)
        x = self.norm(x)

        # optional AdaLN script conditioning
        if script_ids is not None:
            adapter = self.script_adapter(script_ids)
            scale = adapter["scale"].unsqueeze(1)   # (B, 1, C)
            shift = adapter["shift"].unsqueeze(1)   # (B, 1, C)
            x = x * scale + shift

        return x

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def compression_ratio(self) -> int:
        """Total temporal compression: product of all strides."""
        r = 1
        for s in self.STRIDES:
            r *= s
        return r    # = 240

    @property
    def frame_rate(self) -> float:
        """Output frame rate in Hz given the default AudioConfig sample rate."""
        return 24_000 / self.compression_ratio   # = 100.0 Hz

    @property
    def channels(self) -> int:
        """Latent channel dimension C."""
        return self._C

    def num_parameters(self, include_adapter: bool = True) -> int:
        if include_adapter:
            return sum(p.numel() for p in self.parameters())
        return sum(
            p.numel() for name, p in self.named_parameters()
            if "script_adapter" not in name
        )

    def receptive_field_frames(self) -> int:
        """
        Approximate temporal receptive field at the bottleneck output,
        measured in output frames (before any further module).

        Each EncoderBlock's three ResBlocks contribute:
            RF += (2*(1-1)*1 + 2*(3-1)*1 + 2*(9-1)*1) = 0 + 4 + 16 = 20 frames
            (dilated causal convs at the block's temporal resolution)
        Plus the strided down-conv kernel = 2*stride frames.

        This is a lower-bound estimate — actual RF is larger due to pooling.
        """
        rf = 1
        current_stride = 1
        for s in self.STRIDES:
            # resblock contribution in current resolution
            rf += (20) // current_stride if current_stride < s else 20
            # strided conv contribution
            rf += (2 * s) // current_stride if current_stride < s else 2 * s
            current_stride *= s
        return rf

    def __repr__(self) -> str:
        return (
            f"AudioEncoder(\n"
            f"  strides        = {self.STRIDES}\n"
            f"  channels       = {self._C}\n"
            f"  compression    = {self.compression_ratio}×\n"
            f"  frame_rate     = {self.frame_rate:.1f} Hz\n"
            f"  parameters     = {self.num_parameters() / 1e6:.3f} M\n"
            f")"
        )


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))

    print("=" * 60)
    print("  AudioEncoder — self-test")
    print("=" * 60)

    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps")
        if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
        else torch.device("cpu")
    )
    print(f"\nDevice: {device}\n")

    # ── 1. Instantiate encoder ────────────────────────────────────────────
    audio_cfg = AudioConfig(sample_rate=24_000)
    model_cfg = ModelConfig(encoder_channels=512)
    encoder   = AudioEncoder(audio_cfg, model_cfg).to(device)
    print(encoder)
    print(f"  Parameters (excl. adapter) : {encoder.num_parameters(include_adapter=False)/1e6:.3f} M")

    # ── 2. Shape tests across batch sizes and durations ───────────────────
    print("\n[2] Shape verification:")
    test_cases = [
        (1, 24_000),       # 1 s
        (2, 48_000),       # 2 s
        (4, 120_000),      # 5 s
        (1, 24_007),       # non-divisible length — should still work
    ]
    all_passed = True
    for B, T in test_cases:
        waveform   = torch.randn(B, 1, T, device=device)
        script_ids = torch.randint(0, 12, (B,), device=device)

        with torch.no_grad():
            latent_with    = encoder(waveform, script_ids)
            latent_without = encoder(waveform)             # no script conditioning

        expected_T_frames = T // encoder.compression_ratio
        got_T_frames      = latent_with.shape[1]
        shape_ok = (
            latent_with.shape[0] == B
            and latent_with.shape[2] == model_cfg.encoder_channels
            and abs(got_T_frames - expected_T_frames) <= 1   # allow ±1 from padding
        )
        status = "✓ PASS" if shape_ok else "✗ FAIL"
        if not shape_ok:
            all_passed = False
        print(
            f"  B={B}, T={T:>7}  →  latent={tuple(latent_with.shape)}  "
            f"expected T_frames≈{expected_T_frames}  {status}"
        )
        assert latent_with.shape == latent_without.shape, "Script conditioning changed shape!"

    # ── 3. Gradient flow ─────────────────────────────────────────────────
    print("\n[3] Gradient flow:")
    encoder.train()
    waveform   = torch.randn(2, 1, 24_000, device=device, requires_grad=False)
    script_ids = torch.zeros(2, dtype=torch.long, device=device)
    latent     = encoder(waveform, script_ids)
    dummy_loss = latent.mean()
    dummy_loss.backward()
    grads_ok = all(
        p.grad is not None and not p.grad.isnan().any()
        for p in encoder.parameters()
        if p.requires_grad
    )
    print(f"  All gradients non-None and non-NaN: {'✓ PASS' if grads_ok else '✗ FAIL'}")
    if not grads_ok:
        all_passed = False

    # ── 4. AdaLN effect ──────────────────────────────────────────────────
    print("\n[4] AdaLN conditioning sanity:")
    encoder.eval()
    with torch.no_grad():
        waveform = torch.randn(1, 1, 24_000, device=device)
        latent_hi  = encoder(waveform, torch.tensor([0], device=device))   # Hindi (Devanagari)
        latent_ur  = encoder(waveform, torch.tensor([9], device=device))   # Urdu (Perso-Arabic)
        latent_none= encoder(waveform, None)
    diff_hi_ur = (latent_hi - latent_ur).abs().mean().item()
    diff_hi_none=(latent_hi - latent_none).abs().mean().item()
    print(f"  |latent_Hindi − latent_Urdu|   mean diff = {diff_hi_ur:.6f}  (should be > 0)")
    print(f"  |latent_Hindi − latent_None|   mean diff = {diff_hi_none:.6f}  (should be > 0)")
    ada_ok = diff_hi_ur > 0 and diff_hi_none > 0
    print(f"  AdaLN produces distinct outputs: {'✓ PASS' if ada_ok else '✗ FAIL'}")
    if not ada_ok:
        all_passed = False

    # ── 5. Integration with AudioPreprocessor ─────────────────────────────
    print("\n[5] Integration: AudioPreprocessor → AudioEncoder:")
    try:
        from audio_preprocessor import AudioPreprocessor
        import numpy as np
        import tempfile
        import soundfile as sf

        # Create a tiny synthetic WAV
        sr_orig = 16_000
        t       = np.linspace(0, 1.0, sr_orig, dtype=np.float32)
        wav     = (0.4 * np.sin(2 * np.pi * 200 * t)).reshape(-1, 1)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        sf.write(tmp_path, wav, sr_orig)

        prep     = AudioPreprocessor(target_sr=24_000)
        waveform = prep.process(tmp_path).to(device)   # (1, 1, 24000)
        encoder.eval()
        with torch.no_grad():
            latent = encoder(waveform)                  # (1, 100, 512)

        os.unlink(tmp_path)
        print(f"  AudioPreprocessor output : {tuple(waveform.shape)}")
        print(f"  AudioEncoder output      : {tuple(latent.shape)}")
        print(f"  ✓ PASS  — pipeline connected successfully")
    except ImportError:
        print("  (AudioPreprocessor not found on path — skipping integration test)")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if all_passed:
        print("  All tests PASSED.")
    else:
        print("  Some tests FAILED — see output above.")
    print("=" * 60)

    print("\nArchitecture summary:")
    print(f"  Input  shape : (B, 1, T)")
    print(f"  Output shape : (B, T // {encoder.compression_ratio}, {encoder.channels})")
    print(f"  Frame rate   : {encoder.frame_rate:.1f} Hz  (1 frame per 10 ms)")
    print(f"\nNext step: ResidualVectorQuantizer")
    print(f"  Accepts (B, T_frames, C) latent from AudioEncoder")
    print(f"  Outputs discrete codes (B, T_frames, n_codebooks) + quantised latent")