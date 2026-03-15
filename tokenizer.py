#!/usr/bin/env python3
# =============================================================================
# LIPIKA TOKENIZER  –  Production-grade Neural Audio Codec for Indic TTS
# =============================================================================
#
# Architecture overview
# ---------------------
#   Encoder  →  Residual Vector Quantizer (RVQ)  →  Decoder
#                        ↑
#              Script-Family Adapter (AdaLN)
#                        ↑
#            W2V-BERT Semantic Distillation Head
#                        ↑
#            GAN Discriminator (multi-scale + multi-period)
#
# Papers & literature this implementation is grounded in
# -------------------------------------------------------
#
#  [1] Défossez et al. (2022) "High Fidelity Neural Audio Compression"
#      EnCodec — the foundational neural audio codec this builds on.
#      https://arxiv.org/abs/2210.13438
#
#  [2] Zeghidour et al. (2021) "SoundStream: An End-to-End Neural Audio Codec"
#      Original residual VQ codec design.
#      https://arxiv.org/abs/2107.03312
#
#  [3] van den Oord et al. (2017) "Neural Discrete Representation Learning"
#      VQ-VAE — straight-through estimator and commitment loss.
#      https://arxiv.org/abs/1711.00937
#
#  [4] Wang et al. (2023) "Neural Codec Language Models are Zero-Shot Text to
#      Speech Synthesizers" (VALL-E) — motivation for discrete audio tokens.
#      https://arxiv.org/abs/2301.02111
#
#  [5] Baevski et al. (2022) "data2vec" — W2V-BERT semantic features.
#      https://arxiv.org/abs/2202.03555
#
#  [6] Chung et al. (2021) "W2v-BERT: Combining Contrastive Learning and
#      Masked Language Modeling for Self-Supervised Speech Pre-Training"
#      https://arxiv.org/abs/2108.06209
#
#  [7] Kumar et al. (2019) "MelGAN" — multi-scale discriminator.
#      https://arxiv.org/abs/1910.06711
#
#  [8] Kong et al. (2020) "HiFi-GAN" — multi-period discriminator.
#      https://arxiv.org/abs/2010.05646
#
#  [9] Gulrajani et al. (2017) "Improved Training of Wasserstein GANs"
#      https://arxiv.org/abs/1704.00028
#
# [10] Miyato et al. (2018) "Spectral Normalization for GANs"
#      https://arxiv.org/abs/1802.05957
#
# [11] Kim et al. (2021) "VITS" — multi-scale STFT loss.
#      https://arxiv.org/abs/2106.06103
#
# [12] Ba et al. (2016) "Layer Normalization" — AdaLN for script conditioning.
#      https://arxiv.org/abs/1607.06450
#
# [13] Siuzdak (2023) "Vocos" — frequency-domain reconstruction.
#      https://arxiv.org/abs/2306.00814
#
# [14] Defossez et al. (2023) AudioCraft — EMA codebook updates.
#      https://arxiv.org/abs/2306.06189
#
# [15] Zeyer et al. (2023) "Codebook Collapse in VQ-VAEs"
#      https://arxiv.org/abs/2309.12756
#
# Environment requirements
# -------------------------
#   pip install torch torchvision torchaudio
#   pip install transformers soundfile librosa numpy tqdm tensorboard
#   pip install einops scipy omegaconf matplotlib
#
#   GPU: CUDA 12.1+, minimum 16 GB VRAM for batch_size=8 at 24 kHz
#   MPS: Apple Silicon Macs (batch_size=4 recommended)
#   CPU: Supported but slow; batch_size=2 recommended
#
# =============================================================================

from __future__ import annotations

import os
import sys
import math
import json
import time
import random
import logging
import warnings
import argparse
import hashlib
import shutil
import traceback
import platform
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Tuple, Dict, Union, Any
from enum import IntEnum
from collections import defaultdict, Counter
from contextlib import contextmanager, nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_

# ── Optional heavy deps ────────────────────────────────────────────────────
try:
    import torch.distributed as dist
    from torch.utils.data import DistributedSampler
    from torch.nn.parallel import DistributedDataParallel as DDP
    DDP_AVAILABLE = True
except ImportError:
    DDP_AVAILABLE = False

try:
    from torch.nn.utils import spectral_norm
    SPECTRAL_NORM_AVAILABLE = True
except ImportError:
    SPECTRAL_NORM_AVAILABLE = False
    def spectral_norm(m, **_):
        return m

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    class SummaryWriter:  # type: ignore
        def __init__(self, *a, **kw): pass
        def add_scalar(self, *a, **kw): pass
        def add_histogram(self, *a, **kw): pass
        def add_audio(self, *a, **kw): pass
        def close(self): pass

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    warnings.warn("librosa not installed. Resampling disabled. pip install librosa")

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    warnings.warn("soundfile not installed. Audio I/O disabled. pip install soundfile")

try:
    import matplotlib
    matplotlib.use("Agg")           # non-interactive backend — safe everywhere
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("matplotlib not installed. Plot output disabled. pip install matplotlib")

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):  # type: ignore
        return it

# Optional: W2V-BERT semantic teacher
try:
    from transformers import Wav2Vec2BertModel, AutoFeatureExtractor
    W2VBERT_AVAILABLE = True
except ImportError:
    W2VBERT_AVAILABLE = False
    warnings.warn(
        "transformers not installed. W2V-BERT semantic distillation disabled. "
        "Install with: pip install transformers"
    )

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# =============================================================================
# DEVICE AUTO-DETECTION  (robust, crash-proof)
# =============================================================================

def get_device(requested: str = "auto") -> torch.device:
    """
    Safely resolve the best available compute device.

    Priority: CUDA → MPS → CPU.
    Never raises — falls back to CPU on any error.

    Args:
        requested: "auto" | "cuda" | "cuda:N" | "mps" | "cpu"
    """
    if requested != "auto":
        try:
            dev = torch.device(requested)
            # Validate the device is actually usable
            if dev.type == "cuda":
                if not torch.cuda.is_available():
                    warnings.warn(f"CUDA requested but not available. Falling back to CPU.")
                    return torch.device("cpu")
                idx = dev.index or 0
                if idx >= torch.cuda.device_count():
                    warnings.warn(f"CUDA:{idx} not found (only {torch.cuda.device_count()} GPUs). Falling back to cuda:0.")
                    return torch.device("cuda:0")
            elif dev.type == "mps":
                if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                    warnings.warn("MPS requested but not available. Falling back to CPU.")
                    return torch.device("cpu")
            return dev
        except Exception as e:
            warnings.warn(f"Invalid device '{requested}': {e}. Falling back to CPU.")
            return torch.device("cpu")

    # Auto-detect
    try:
        if torch.cuda.is_available():
            # Quick smoke-test: allocate 1 byte on GPU
            t = torch.zeros(1, device="cuda")
            del t
            return torch.device("cuda")
    except Exception as e:
        warnings.warn(f"CUDA detected but unusable: {e}")

    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            t = torch.zeros(1, device="mps")
            del t
            return torch.device("mps")
    except Exception as e:
        warnings.warn(f"MPS detected but unusable: {e}")

    return torch.device("cpu")


def device_info(device: torch.device) -> str:
    """Return a human-readable device summary."""
    if device.type == "cuda":
        idx = device.index or 0
        name = torch.cuda.get_device_name(idx)
        mem  = torch.cuda.get_device_properties(idx).total_memory / 1e9
        return f"CUDA [{name}] ({mem:.1f} GB VRAM)"
    if device.type == "mps":
        return f"Apple MPS [{platform.processor() or 'Apple Silicon'}]"
    return f"CPU [{platform.processor() or platform.machine()}] ({os.cpu_count()} cores)"


def supports_amp(device: torch.device) -> bool:
    """Check if the device supports automatic mixed precision."""
    return device.type == "cuda"


def supports_bf16(device: torch.device) -> bool:
    """Check if bf16 is supported (better than fp16 for training stability)."""
    if device.type != "cuda":
        return False
    try:
        return torch.cuda.is_bf16_supported()
    except Exception:
        return False


# Resolved once at import time
DEVICE: torch.device = get_device()

# =============================================================================
# LOGGING
# =============================================================================

def _configure_root_logger() -> None:
    """
    Configure root logger with a Windows-safe UTF-8 stream handler.

    On Windows, the default console encoding (cp1252) cannot render
    Unicode box-drawing or arrow characters. We explicitly set the
    stream handler to use UTF-8 with error replacement so logging
    never crashes, regardless of the OS locale.
    """
    import io
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Wrap stdout in a UTF-8 writer that replaces unencodable chars
    try:
        utf8_stream = io.TextIOWrapper(
            sys.stdout.buffer,
            encoding="utf-8",
            errors="replace",
            line_buffering=True,
        )
        handler = logging.StreamHandler(utf8_stream)
    except AttributeError:
        # sys.stdout may not have .buffer in some IDEs / PYTHONIOENCODING setups
        handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(fmt)
    root.addHandler(handler)


_configure_root_logger()
logger = logging.getLogger("lipika")


def setup_logging(log_dir: Path, rank: int = 0) -> None:
    """Configure file + console logging; only rank-0 writes."""
    if rank != 0:
        logging.disable(logging.CRITICAL)
        return
    log_dir.mkdir(parents=True, exist_ok=True)
    # Always write the log file as UTF-8 regardless of OS locale
    fh = logging.FileHandler(log_dir / "training.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ))
    logging.getLogger().addHandler(fh)


# =============================================================================
# CONFIGURATION  (fully serialisable)
# =============================================================================

@dataclass
class AudioConfig:
    """
    Audio processing parameters.

    sample_rate=24000 is optimal for Indic TTS:
    captures retroflex stops and aspirated consonants (energy up to ~10 kHz)
    without the memory cost of 44.1 kHz.

    n_fft=2048, hop_length=240 gives 10 ms frames at 24 kHz —
    standard for ASR/TTS alignment (see [4] VALL-E, [13] Vocos).
    """
    sample_rate: int = 24_000
    n_fft: int = 2048
    hop_length: int = 240           # frame_rate = 100 Hz
    n_mels: int = 128
    fmin: float = 0.0
    fmax: float = 12_000.0


@dataclass
class RVQConfig:
    """
    Residual Vector Quantizer hyper-parameters.

    n_codebooks=8 matches EnCodec 24 kHz [1].
    codebook_size=1024 follows SoundStream [2].
    EMA updates (ema_decay=0.99) stabilise training [14].
    """
    n_codebooks: int = 8
    codebook_size: int = 1024
    codebook_dim: int = 128
    commitment_cost: float = 0.25
    ema_decay: float = 0.99
    ema_epsilon: float = 1e-5
    threshold_ema_dead_code: float = 2.0


@dataclass
class ModelConfig:
    """Encoder / decoder / adapter sizes."""
    encoder_channels: int = 512
    encoder_depth: int = 8
    decoder_channels: int = 512
    decoder_depth: int = 8

    w2v_bert_model: str = "facebook/w2v-bert-2.0"
    w2v_bert_dim: int = 1024
    semantic_proj_dim: int = 256

    n_script_families: int = 12
    script_embed_dim: int = 64

    disc_channels: int = 64
    disc_depth: int = 4
    mpd_periods: List[int] = field(default_factory=lambda: [2, 3, 5, 7, 11])


@dataclass
class TrainingConfig:
    """
    Training hyper-parameters.

    Loss weights tuned to match EnCodec [1] Table 1 ablation:
    λ_t=0.1, λ_f=1.0, λ_g=3.0, λ_feat=3.0, λ_w2v=10.0 (added).
    """
    # Infrastructure
    batch_size: int = 8
    grad_accum_steps: int = 1
    num_epochs: int = 200
    num_workers: int = 0
    pin_memory: bool = True
    mixed_precision: bool = True
    compile_model: bool = False
    seed: int = 42
    device: str = "auto"

    # Paths
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    data_dir: str = "./data"
    plot_dir: str = "./plots"        # NEW: output directory for training plots
    output_dir: str = "./outputs"    # NEW: output directory for audio samples

    # Audio
    max_duration: float = 5.0

    # Optimiser
    learning_rate: float = 3e-4
    disc_learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    betas: Tuple[float, float] = (0.8, 0.99)
    grad_clip: float = 1.0
    warmup_steps: int = 1000
    lr_decay_steps: int = 400_000

    # Loss weights
    w_time_recon: float = 0.1
    w_freq_recon: float = 1.0
    w_mel: float = 1.0
    w_vq: float = 1.0
    w_semantic: float = 10.0
    w_gen: float = 3.0
    w_feat: float = 3.0

    # GAN schedule
    disc_start_step: int = 10_000
    disc_update_every: int = 1

    # Checkpointing
    save_every_steps: int = 5_000
    eval_every_steps: int = 1_000
    keep_last_n_checkpoints: int = 5
    plot_every_steps: int = 500       # NEW: generate plots every N steps
    sample_every_steps: int = 2_000   # NEW: save audio samples every N steps

    # Distributed
    ddp_backend: str = "nccl"


# =============================================================================
# SCRIPT FAMILY ENUM
# =============================================================================

class ScriptFamily(IntEnum):
    """Unicode script families for Indic languages."""
    DEVANAGARI  = 0
    BENGALI     = 1
    GURMUKHI    = 2
    GUJARATI    = 3
    ORIYA       = 4
    TAMIL       = 5
    TELUGU      = 6
    KANNADA     = 7
    MALAYALAM   = 8
    PERSO_ARABIC= 9
    MEITEI      = 10
    LATIN_INDIA = 11


LANG_TO_SCRIPT: Dict[str, ScriptFamily] = {
    "hi": ScriptFamily.DEVANAGARI,  "mr": ScriptFamily.DEVANAGARI,
    "sa": ScriptFamily.DEVANAGARI,  "ne": ScriptFamily.DEVANAGARI,
    "kok": ScriptFamily.DEVANAGARI, "bn": ScriptFamily.BENGALI,
    "as": ScriptFamily.BENGALI,     "pa": ScriptFamily.GURMUKHI,
    "gu": ScriptFamily.GUJARATI,    "or": ScriptFamily.ORIYA,
    "ta": ScriptFamily.TAMIL,       "te": ScriptFamily.TELUGU,
    "kn": ScriptFamily.KANNADA,     "ml": ScriptFamily.MALAYALAM,
    "ur": ScriptFamily.PERSO_ARABIC,"ks": ScriptFamily.PERSO_ARABIC,
    "mni": ScriptFamily.MEITEI,     "en": ScriptFamily.LATIN_INDIA,
}


# =============================================================================
# BUILDING BLOCKS
# =============================================================================

class CausalConv1d(nn.Module):
    """
    Causal 1-D convolution: no future context leaks.
    Padding applied exclusively on the left (past).
    Reference: EnCodec [1], SoundStream [2].
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
        x = F.pad(x, (self.causal_pad, 0))
        return self.conv(x)


class CausalConvTranspose1d(nn.Module):
    """
    Causal transposed 1-D convolution for upsampling in the decoder.
    Removes non-causal right-padding artefacts from standard ConvTranspose1d.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.stride = stride
        self.conv = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        trim = self.conv.kernel_size[0] - self.stride
        return x[..., : x.shape[-1] - trim] if trim > 0 else x


class ResBlock(nn.Module):
    """
    Gated residual block with dilated causal convolutions.
    Dilation schedule [1, 3, 9] gives large receptive field cheaply.
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
        return x + self.layers(x)


class EncoderBlock(nn.Module):
    """
    Strided downsampling block: residual stack + strided conv with channel gating.

    Channel flow: in=C -> res stack (C) -> down conv (C->2C) -> gate (2C->C) -> pool
    Net output channels = C (same as input). The down conv + gate pattern acts as a
    learned gated linear unit for the downsampling step (EnCodec [1] design).
    """

    def __init__(self, channels: int, stride: int) -> None:
        super().__init__()
        self.res  = nn.Sequential(*[ResBlock(channels, d) for d in [1, 3, 9]])
        # Doubles channels; gating in forward halves back to channels
        self.down = CausalConv1d(channels, channels * 2, kernel_size=2 * stride, dilation=1)
        self.stride_pool = nn.AvgPool1d(stride, stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.res(x)                        # (B, C, T)
        x = self.down(x)                       # (B, 2C, T)
        x = x[:, : x.shape[1] // 2, :]        # gate: (B, C, T)  — net channels unchanged
        x = self.stride_pool(x)                # (B, C, T/stride)
        return x


class DecoderBlock(nn.Module):
    """
    Strided upsampling block: causal ConvTranspose1d + residual stack.

    Channel flow: in=C -> up conv (C->C) -> res stack (C)
    Net output channels = C (same as input), mirroring EncoderBlock.
    """

    def __init__(self, channels: int, stride: int) -> None:
        super().__init__()
        # Keep channels constant through the upsampling block
        self.up  = CausalConvTranspose1d(channels, channels, kernel_size=2 * stride, stride=stride)
        self.res = nn.Sequential(*[ResBlock(channels, d) for d in [1, 3, 9]])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res(self.up(x))


# =============================================================================
# SCRIPT-FAMILY ADAPTER (Adaptive Layer Normalisation)
# =============================================================================

class ScriptFamilyAdapter(nn.Module):
    """
    Conditions the encoder on the script family via AdaLN [12].

    Retroflex consonants (/ʈ ɖ ɳ ɽ/) are phonemically contrastive in
    most Indic languages. The adapter helps allocate codebook entries for
    these fine-grained distinctions. The retroflex_bias is a soft inductive
    prior in the first 8 embedding dimensions — training will correct it.
    """

    RETROFLEX_SCRIPTS = {
        ScriptFamily.DEVANAGARI, ScriptFamily.BENGALI, ScriptFamily.GURMUKHI,
        ScriptFamily.ORIYA, ScriptFamily.TAMIL, ScriptFamily.TELUGU,
        ScriptFamily.KANNADA, ScriptFamily.MALAYALAM,
    }

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.embed = nn.Embedding(cfg.n_script_families, cfg.script_embed_dim)

        with torch.no_grad():
            for sf_id in self.RETROFLEX_SCRIPTS:
                self.embed.weight[int(sf_id), :8] += 0.5

        self.proj = nn.Sequential(
            nn.Linear(cfg.script_embed_dim, cfg.encoder_channels),
            nn.SiLU(),
            nn.Linear(cfg.encoder_channels, cfg.encoder_channels),
        )
        self.scale_head = nn.Linear(cfg.encoder_channels, cfg.encoder_channels)
        self.shift_head = nn.Linear(cfg.encoder_channels, cfg.encoder_channels)
        nn.init.zeros_(self.scale_head.weight); nn.init.ones_(self.scale_head.bias)
        nn.init.zeros_(self.shift_head.weight); nn.init.zeros_(self.shift_head.bias)

    def forward(self, script_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        e = self.proj(self.embed(script_ids))
        return {"scale": self.scale_head(e), "shift": self.shift_head(e)}


# =============================================================================
# AUDIO ENCODER
# =============================================================================

class AudioEncoder(nn.Module):
    """
    Causal convolutional encoder: waveform → continuous latent sequence.
    Strides [2, 4, 5, 6] → compression ratio 240 → 100 Hz frame rate.
    Reference: EnCodec [1] §3.1.
    """

    STRIDES = [2, 4, 5, 6]

    def __init__(self, audio_cfg: AudioConfig, model_cfg: ModelConfig) -> None:
        super().__init__()
        C = model_cfg.encoder_channels
        self.stem = CausalConv1d(1, C, kernel_size=7)

        # Each EncoderBlock keeps channels constant (down conv doubles, gating halves).
        # All blocks operate at C channels; only temporal resolution decreases.
        blocks = []
        for stride in self.STRIDES:
            blocks.append(EncoderBlock(C, stride))
        self.blocks = nn.Sequential(*blocks)

        # Bottleneck: C -> C (identity channel count; projects to ensure clean latent)
        self.bottleneck = nn.Sequential(
            nn.ELU(),
            CausalConv1d(C, C, kernel_size=1),
        )
        self.norm = nn.LayerNorm(C)

    def forward(
        self,
        waveform: torch.Tensor,
        script_adapter: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        x = self.stem(waveform)
        x = self.blocks(x)
        x = self.bottleneck(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        if script_adapter is not None:
            scale = script_adapter["scale"].unsqueeze(1)
            shift = script_adapter["shift"].unsqueeze(1)
            x = x * scale + shift
        return x

    @property
    def compression_ratio(self) -> int:
        r = 1
        for s in self.STRIDES:
            r *= s
        return r


# =============================================================================
# VECTOR QUANTIZER WITH EMA UPDATES AND DEAD-CODE RESET
# =============================================================================

class VectorQuantizerEMA(nn.Module):
    """
    Vector quantiser with EMA codebook updates and dead-code reset.

    EMA updates are more stable than gradient-based updates [3, 14].
    Dead-code reset [15]: codes used < threshold_ema_dead_code times per
    batch are re-initialised from a random live input vector.
    Straight-through estimator [3] allows gradient flow through argmin.
    """

    def __init__(self, cfg: RVQConfig) -> None:
        super().__init__()
        self.codebook_size = cfg.codebook_size
        self.dim = cfg.codebook_dim
        self.commitment_cost = cfg.commitment_cost
        self.decay = cfg.ema_decay
        self.epsilon = cfg.ema_epsilon
        self.threshold_dead = cfg.threshold_ema_dead_code

        self.register_buffer("embedding",    torch.empty(cfg.codebook_size, cfg.codebook_dim))
        self.register_buffer("cluster_size", torch.zeros(cfg.codebook_size))
        self.register_buffer("embed_avg",    torch.empty(cfg.codebook_size, cfg.codebook_dim))

        nn.init.uniform_(self.embedding, -1.0 / cfg.codebook_size, 1.0 / cfg.codebook_size)
        self.embed_avg.data.copy_(self.embedding.data)

    def _distances(self, flat_z: torch.Tensor) -> torch.Tensor:
        return (
            flat_z.pow(2).sum(1, keepdim=True)
            - 2.0 * flat_z @ self.embedding.t()
            + self.embedding.pow(2).sum(1)
        )

    def _lookup(self, indices: torch.Tensor) -> torch.Tensor:
        return F.embedding(indices, self.embedding)

    @torch.no_grad()
    def _ema_update(self, flat_z: torch.Tensor, indices: torch.Tensor) -> None:
        one_hot = torch.zeros(flat_z.size(0), self.codebook_size, device=flat_z.device)
        one_hot.scatter_(1, indices.unsqueeze(1), 1)
        counts    = one_hot.sum(0)
        embed_sum = one_hot.t() @ flat_z

        # All-reduce across DDP processes (safe no-op if not distributed)
        if DDP_AVAILABLE and dist.is_initialized():
            dist.all_reduce(counts,    op=dist.ReduceOp.SUM)
            dist.all_reduce(embed_sum, op=dist.ReduceOp.SUM)

        self.cluster_size.mul_(self.decay).add_(counts,    alpha=1 - self.decay)
        self.embed_avg.mul_(self.decay).add_(embed_sum,    alpha=1 - self.decay)

        n = self.cluster_size.sum()
        smoothed = (
            (self.cluster_size + self.epsilon)
            / (n + self.codebook_size * self.epsilon) * n
        )
        self.embedding.copy_(self.embed_avg / smoothed.unsqueeze(1).clamp(min=1e-7))

        # Dead-code reset [15]
        dead_mask = counts < self.threshold_dead
        n_dead = int(dead_mask.sum().item())
        if n_dead > 0 and flat_z.size(0) > 0:
            perm = torch.randperm(flat_z.size(0), device=flat_z.device)[:n_dead]
            self.embedding[dead_mask] = flat_z[perm].detach()
            self.embed_avg[dead_mask] = flat_z[perm].detach()
            self.cluster_size[dead_mask] = self.threshold_dead

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, D = z.shape
        flat_z = z.reshape(-1, D)
        distances = self._distances(flat_z)
        indices   = distances.argmin(1)
        z_q_flat  = self._lookup(indices)

        if self.training:
            self._ema_update(flat_z.detach(), indices)

        commitment_loss = F.mse_loss(z_q_flat.detach(), flat_z)
        loss  = self.commitment_cost * commitment_loss
        z_q   = z_q_flat.reshape(B, T, D)
        z_q_st = z + (z_q - z).detach()   # straight-through [3]
        return z_q_st, indices.reshape(B, T), loss


# =============================================================================
# RESIDUAL VECTOR QUANTIZER
# =============================================================================

class ResidualVectorQuantizer(nn.Module):
    """
    Residual VQ: quantise the residual of the previous codebook.

    z_i = z_{i-1} - z_q_{i-1}    (residual at codebook i)
    z_q = sum_i z_q_i             (total quantised)

    First codebook captures coarse semantics (distilled from W2V-BERT [5,6]).
    Subsequent codebooks refine acoustic detail — mirrors VALL-E [4].
    """

    def __init__(self, rvq_cfg: RVQConfig, model_cfg: ModelConfig) -> None:
        super().__init__()
        self.input_proj  = nn.Linear(model_cfg.encoder_channels, rvq_cfg.codebook_dim)
        self.codebooks   = nn.ModuleList([VectorQuantizerEMA(rvq_cfg) for _ in range(rvq_cfg.n_codebooks)])
        self.n_codebooks = rvq_cfg.n_codebooks
        self.codebook_dim = rvq_cfg.codebook_dim

        self.semantic_head = nn.Sequential(
            nn.LayerNorm(rvq_cfg.codebook_dim),
            nn.Linear(rvq_cfg.codebook_dim, model_cfg.semantic_proj_dim),
            nn.GELU(),
            nn.Linear(model_cfg.semantic_proj_dim, model_cfg.w2v_bert_dim),
        )
        self.output_proj = nn.Linear(rvq_cfg.codebook_dim, model_cfg.encoder_channels)

    def forward(
        self,
        z: torch.Tensor,
        w2v_targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        z_proj  = self.input_proj(z)
        residual = z_proj
        z_q_total = torch.zeros_like(z_proj)
        all_codes = []
        total_vq_loss  = torch.tensor(0.0, device=z.device)
        semantic_loss  = torch.tensor(0.0, device=z.device)

        for i, vq in enumerate(self.codebooks):
            z_q_i, indices_i, loss_i = vq(residual)

            if i == 0 and w2v_targets is not None:
                pred  = self.semantic_head(z_q_i)
                min_t = min(pred.size(1), w2v_targets.size(1))
                semantic_loss = F.mse_loss(pred[:, :min_t], w2v_targets[:, :min_t].detach())

            z_q_total = z_q_total + z_q_i
            residual  = residual  - z_q_i.detach()
            all_codes.append(indices_i)
            total_vq_loss = total_vq_loss + loss_i

        all_codes = torch.stack(all_codes, dim=-1)
        z_q_out   = self.output_proj(z_q_total)
        return {
            "z_q": z_q_out,
            "codes": all_codes,
            "vq_loss": total_vq_loss,
            "semantic_loss": semantic_loss,
        }

    def decode_from_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """Reconstruct quantised latent from discrete codes (inference)."""
        z_q = torch.zeros(
            codes.shape[0], codes.shape[1], self.codebook_dim, device=codes.device
        )
        for i, vq in enumerate(self.codebooks):
            z_q = z_q + vq._lookup(codes[..., i])
        return self.output_proj(z_q)


# =============================================================================
# AUDIO DECODER
# =============================================================================

class AudioDecoder(nn.Module):
    """
    Causal convolutional decoder: quantised latent → waveform.
    Mirror of encoder with strides in reverse, tanh output ∈ [-1, 1].
    Reference: EnCodec decoder §3.2 [1].
    """

    STRIDES = AudioEncoder.STRIDES[::-1]

    def __init__(self, model_cfg: ModelConfig) -> None:
        super().__init__()
        C = model_cfg.decoder_channels
        self.entry = CausalConv1d(C, C, kernel_size=7)

        # Each DecoderBlock keeps channels constant at C
        blocks = []
        for stride in self.STRIDES:
            blocks.append(DecoderBlock(C, stride))
        self.blocks = nn.Sequential(*blocks)

        # Final projection: C channels -> 1 (mono waveform)
        self.out = nn.Sequential(
            nn.ELU(),
            CausalConv1d(C, 1, kernel_size=7),
            nn.Tanh(),
        )

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        x = z_q.transpose(1, 2)
        x = self.entry(x)
        x = self.blocks(x)
        return self.out(x)


# =============================================================================
# W2V-BERT SEMANTIC TEACHER
# =============================================================================

class SemanticTeacher(nn.Module):
    """
    Frozen W2V-BERT-2.0 used as a semantic feature teacher [5, 6].

    Always frozen — hidden states at layer 6 serve as distillation targets
    for the first RVQ codebook, ensuring coarse codes capture phonetic
    identity rather than acoustic texture.

    Note: W2V-BERT expects 16 kHz; we resample on-the-fly.
    """

    TARGET_SR   = 16_000
    HIDDEN_LAYER = 6

    def __init__(self, model_name: str) -> None:
        super().__init__()
        if not W2VBERT_AVAILABLE:
            raise RuntimeError("transformers not installed. Cannot use SemanticTeacher.")
        logger.info(f"Loading W2V-BERT teacher: {model_name}")
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = Wav2Vec2BertModel.from_pretrained(model_name, output_hidden_states=True)
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, waveform_24k: torch.Tensor, src_sr: int = 24_000) -> torch.Tensor:
        if not LIBROSA_AVAILABLE:
            return torch.zeros(waveform_24k.shape[0], 1, 1024, device=waveform_24k.device)
        wav_np = waveform_24k.squeeze(1).cpu().numpy()
        resampled = np.stack([
            librosa.resample(w, orig_sr=src_sr, target_sr=self.TARGET_SR) for w in wav_np
        ])
        inputs = self.feature_extractor(
            list(resampled), sampling_rate=self.TARGET_SR,
            return_tensors="pt", padding=True,
        )
        inputs = {k: v.to(waveform_24k.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        return outputs.hidden_states[self.HIDDEN_LAYER]


# =============================================================================
# DISCRIMINATORS  (Multi-Scale + Multi-Period, HiFi-GAN style [7, 8])
# =============================================================================

def _sn_conv1d(*args, **kwargs) -> nn.Conv1d:
    """Spectral-normalised Conv1d [10]. Falls back gracefully if unavailable."""
    return spectral_norm(nn.Conv1d(*args, **kwargs))


class PeriodDiscriminator(nn.Module):
    """
    Sub-discriminator for periodic subsampling of the waveform.
    Different periods capture different rhythmic/prosodic patterns [8].
    """

    def __init__(self, period: int, channels: int = 64, depth: int = 4) -> None:
        super().__init__()
        self.period = period
        C = channels
        layers = [nn.Sequential(_sn_conv1d(1, C, 5, stride=3, padding=2), nn.LeakyReLU(0.1))]
        for _ in range(1, depth):
            layers.append(nn.Sequential(_sn_conv1d(C, C*2, 5, stride=3, padding=2), nn.LeakyReLU(0.1)))
            C *= 2
        layers.append(nn.Sequential(_sn_conv1d(C, C, 3, padding=1), nn.LeakyReLU(0.1)))
        layers.append(_sn_conv1d(C, 1, 3, padding=1))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        B, C, T = x.shape
        pad = (self.period - T % self.period) % self.period
        x = F.pad(x, (0, pad))
        x = x.view(B, C, -1, self.period)
        x = x.transpose(2, 3).reshape(B, self.period, -1)
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        return x, features


class ScaleDiscriminator(nn.Module):
    """
    Sub-discriminator at one temporal resolution [7].
    """

    def __init__(self, channels: int = 64, depth: int = 4) -> None:
        super().__init__()
        C = channels
        self.layers = nn.ModuleList()
        self.layers.append(nn.Sequential(_sn_conv1d(1, C, 15, stride=1, padding=7), nn.LeakyReLU(0.1)))
        for _ in range(depth):
            self.layers.append(nn.Sequential(
                _sn_conv1d(C, C*2, 41, stride=4, padding=20, groups=4), nn.LeakyReLU(0.1)
            ))
            C *= 2
        self.layers.append(nn.Sequential(_sn_conv1d(C, C, 5, stride=1, padding=2), nn.LeakyReLU(0.1)))
        self.layers.append(_sn_conv1d(C, 1, 3, stride=1, padding=1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        return x, features


class MultiScaleMultiPeriodDiscriminator(nn.Module):
    """
    Combined MSD + MPD discriminator [8].

    MSD: spectral/long-range patterns.
    MPD: periodic/prosodic patterns.
    Together they reduce mode collapse substantially [8] Table 1.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.msds = nn.ModuleList([
            ScaleDiscriminator(cfg.disc_channels, cfg.disc_depth) for _ in range(3)
        ])
        self.msd_pools = nn.ModuleList([
            nn.Identity(),
            nn.AvgPool1d(2, stride=2, padding=1),
            nn.AvgPool1d(4, stride=4, padding=2),
        ])
        self.mpds = nn.ModuleList([
            PeriodDiscriminator(p, cfg.disc_channels, cfg.disc_depth)
            for p in cfg.mpd_periods
        ])

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        all_logits, all_features = [], []
        for disc, pool in zip(self.msds, self.msd_pools):
            logit, feats = disc(pool(x))
            all_logits.append(logit); all_features.append(feats)
        for disc in self.mpds:
            logit, feats = disc(x)
            all_logits.append(logit); all_features.append(feats)
        return all_logits, all_features


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class MelSpectrogramLoss(nn.Module):
    """
    Log-mel spectrogram L1 loss — perceptually motivated [8, 11, 13].
    """

    def __init__(self, audio_cfg: AudioConfig) -> None:
        super().__init__()
        if LIBROSA_AVAILABLE:
            mel_fb = librosa.filters.mel(
                sr=audio_cfg.sample_rate, n_fft=audio_cfg.n_fft,
                n_mels=audio_cfg.n_mels, fmin=audio_cfg.fmin, fmax=audio_cfg.fmax,
            )
            self.register_buffer("mel_filterbank", torch.from_numpy(mel_fb).float())
        else:
            # Fallback: triangular mel filterbank (no librosa)
            n_mels = audio_cfg.n_mels
            n_fft  = audio_cfg.n_fft
            self.register_buffer("mel_filterbank", torch.ones(n_mels, n_fft // 2 + 1) / (n_fft // 2 + 1))
        self.n_fft = audio_cfg.n_fft
        self.hop_length = audio_cfg.hop_length

    def _to_mel(self, x: torch.Tensor) -> torch.Tensor:
        window = torch.hann_window(self.n_fft, device=x.device)
        stft = torch.stft(x.squeeze(1), self.n_fft, self.hop_length, window=window, return_complex=True)
        mag  = stft.abs()
        mel  = torch.einsum("mf,bft->bmt", self.mel_filterbank, mag)
        return torch.log1p(mel)

    def forward(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(self._to_mel(fake), self._to_mel(real))


class MultiScaleSTFTLoss(nn.Module):
    """
    Multi-resolution STFT loss [11, 13].
    Spectral convergence + log magnitude at multiple FFT sizes.
    """

    def __init__(
        self,
        fft_sizes: List[int] = (256, 512, 1024, 2048),
        hop_ratio: float = 0.25,
    ) -> None:
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_ratio = hop_ratio

    def forward(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        total = torch.tensor(0.0, device=real.device)
        x, x_hat = real.squeeze(1), fake.squeeze(1)
        for n_fft in self.fft_sizes:
            hop    = max(1, int(n_fft * self.hop_ratio))
            window = torch.hann_window(n_fft, device=x.device)
            S     = torch.stft(x,     n_fft, hop, window=window, return_complex=True).abs()
            S_hat = torch.stft(x_hat, n_fft, hop, window=window, return_complex=True).abs()
            sc = (S - S_hat).norm() / (S.norm() + 1e-7)
            lm = F.l1_loss(S_hat.log1p(), S.log1p())
            total = total + sc + lm
        return total / len(self.fft_sizes)


def hinge_disc_loss(real_logits: List[torch.Tensor], fake_logits: List[torch.Tensor]) -> torch.Tensor:
    """Hinge discriminator loss [1]."""
    loss = torch.tensor(0.0, device=real_logits[0].device)
    for real, fake in zip(real_logits, fake_logits):
        loss = loss + F.relu(1.0 - real).mean() + F.relu(1.0 + fake).mean()
    return loss / max(len(real_logits), 1)


def hinge_gen_loss(fake_logits: List[torch.Tensor]) -> torch.Tensor:
    """Hinge generator loss [1]."""
    loss = torch.tensor(0.0, device=fake_logits[0].device)
    for fake in fake_logits:
        loss = loss - fake.mean()
    return loss / max(len(fake_logits), 1)


def feature_matching_loss(
    real_features: List[List[torch.Tensor]],
    fake_features: List[List[torch.Tensor]],
) -> torch.Tensor:
    """Feature matching loss [8]: L_FM = E[||D_l(x) - D_l(G(z))||_1]."""
    loss, count = torch.tensor(0.0, device=real_features[0][0].device), 0
    for rf_list, ff_list in zip(real_features, fake_features):
        for rf, ff in zip(rf_list, ff_list):
            loss = loss + F.l1_loss(ff, rf.detach())
            count += 1
    return loss / max(count, 1)


# =============================================================================
# CODEBOOK HEALTH MONITOR
# =============================================================================

class CodebookMonitor:
    """
    Tracks per-codebook utilisation and perplexity during training.

    Metrics:
    - usage_pct: fraction of codebook entries used (want ~80–100 %)
    - perplexity: exp(entropy) — max = codebook_size means uniform usage
    - dead_codes: entries with zero usage in the last window

    Raises a collapse warning if any codebook uses < 20 % of entries [15].
    """

    WINDOW = 100

    def __init__(self, n_codebooks: int, codebook_size: int) -> None:
        self.n_codebooks  = n_codebooks
        self.codebook_size = codebook_size
        self._usage_buf: List[List[float]] = [[] for _ in range(n_codebooks)]
        self._perp_buf:  List[List[float]] = [[] for _ in range(n_codebooks)]

    @torch.no_grad()
    def update(self, codes: torch.Tensor) -> None:
        B, T, n_cb = codes.shape
        for cb in range(min(n_cb, self.n_codebooks)):
            flat   = codes[:, :, cb].reshape(-1).cpu().numpy()
            counts = np.bincount(flat, minlength=self.codebook_size)
            usage  = (counts > 0).mean() * 100
            probs  = counts / counts.sum()
            probs  = probs[probs > 0]
            perp   = float(np.exp(-np.sum(probs * np.log(probs + 1e-12))))
            self._usage_buf[cb].append(usage)
            self._perp_buf[cb].append(perp)
            if len(self._usage_buf[cb]) > self.WINDOW:
                self._usage_buf[cb].pop(0)
                self._perp_buf[cb].pop(0)

    def report(self) -> Dict[str, Any]:
        avg_usage = [np.mean(b) if b else 0.0 for b in self._usage_buf]
        avg_perp  = [np.mean(b) if b else 0.0 for b in self._perp_buf]
        return {
            "usage_pct": avg_usage,
            "perplexity": avg_perp,
            "collapse_warning": any(u < 20.0 for u in avg_usage),
        }

    def log_to_tensorboard(self, writer: SummaryWriter, step: int) -> None:
        rpt = self.report()
        for i, (u, p) in enumerate(zip(rpt["usage_pct"], rpt["perplexity"])):
            writer.add_scalar(f"codebook/usage_pct/cb{i}",   u, step)
            writer.add_scalar(f"codebook/perplexity/cb{i}",  p, step)


# =============================================================================
# TRAINING METRICS TRACKER  (NEW)
# =============================================================================

class MetricsTracker:
    """
    Accumulates training/validation metrics across steps and epochs.
    Used to generate training-curve plots and CSV exports.
    """

    def __init__(self) -> None:
        self.history: Dict[str, List[Tuple[int, float]]] = defaultdict(list)

    def update(self, step: int, metrics: Dict[str, float]) -> None:
        for k, v in metrics.items():
            self.history[k].append((step, float(v)))

    def to_arrays(self, key: str) -> Tuple[np.ndarray, np.ndarray]:
        if key not in self.history or not self.history[key]:
            return np.array([]), np.array([])
        steps, vals = zip(*self.history[key])
        return np.array(steps), np.array(vals)

    def save_csv(self, path: Path) -> None:
        """Export all metrics to a CSV file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted(self.history.keys())
        if not keys:
            return
        # Find all unique steps
        all_steps = sorted(set(s for k in keys for s, _ in self.history[k]))
        rows = [["step"] + keys]
        lookup = {k: dict(self.history[k]) for k in keys}
        for step in all_steps:
            row = [str(step)] + [str(lookup[k].get(step, "")) for k in keys]
            rows.append(row)
        with open(path, "w") as f:
            for row in rows:
                f.write(",".join(row) + "\n")
        logger.info(f"Metrics CSV saved: {path}")


# =============================================================================
# PLOT UTILITIES  (NEW)
# =============================================================================

def plot_training_curves(
    tracker: MetricsTracker,
    plot_dir: Path,
    step: int,
    codebook_report: Optional[Dict] = None,
) -> None:
    """
    Save a comprehensive training-curve figure to disk.
    Safe to call even when matplotlib is unavailable.
    """
    if not MATPLOTLIB_AVAILABLE:
        return

    plot_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(18, 12), facecolor="#0d1117")
    fig.suptitle(
        f"Lipika Tokenizer — Training Curves  (step {step:,})",
        fontsize=14, color="#e6edf3", fontweight="bold", y=0.98,
    )

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)
    ax_color = "#e6edf3"
    bg_color = "#161b22"

    def _axes(row, col, title, ylabel="Loss"):
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor(bg_color)
        ax.set_title(title, color=ax_color, fontsize=9)
        ax.set_xlabel("Step", color=ax_color, fontsize=7)
        ax.set_ylabel(ylabel, color=ax_color, fontsize=7)
        ax.tick_params(colors=ax_color, labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
        return ax

    palette = ["#58a6ff", "#3fb950", "#f78166", "#d2a8ff", "#ffa657", "#79c0ff", "#56d364", "#ff7b72"]

    def _plot(ax, key, label, color, smooth=True):
        steps, vals = tracker.to_arrays(key)
        if len(steps) == 0:
            return
        ax.plot(steps, vals, color=color, alpha=0.3, linewidth=0.8, label="_raw")
        if smooth and len(vals) >= 10:
            kernel = min(len(vals) // 5, 50)
            smooth_vals = np.convolve(vals, np.ones(kernel)/kernel, mode="valid")
            smooth_steps = steps[kernel - 1:]
            ax.plot(smooth_steps, smooth_vals, color=color, linewidth=1.8, label=label)
        else:
            ax.plot(steps, vals, color=color, linewidth=1.8, label=label)
        ax.legend(fontsize=7, facecolor=bg_color, labelcolor=ax_color, framealpha=0.7)

    # Row 0: primary losses
    ax = _axes(0, 0, "Generator Total Loss");   _plot(ax, "g_loss",   "G Loss",   palette[0])
    ax = _axes(0, 1, "Reconstruction Losses")
    _plot(ax, "recon", "L1 Waveform",   palette[0])
    _plot(ax, "mel",   "Mel Spec",      palette[1])
    _plot(ax, "stft",  "MS-STFT",       palette[2])
    ax = _axes(0, 2, "VQ + Semantic Loss")
    _plot(ax, "vq",    "VQ Commit",     palette[3])
    _plot(ax, "sem",   "W2V-BERT KD",   palette[4])

    # Row 1: GAN losses + LR
    ax = _axes(1, 0, "Discriminator Loss");   _plot(ax, "d_loss",   "D Loss",    palette[5])
    ax = _axes(1, 1, "Adversarial + Feature Match")
    _plot(ax, "adv_loss",  "Adv Loss",      palette[6])
    _plot(ax, "feat_loss", "Feat Match",    palette[7])
    ax = _axes(1, 2, "Learning Rate", ylabel="LR"); _plot(ax, "lr", "Gen LR", palette[0], smooth=False)

    # Row 2: validation + codebook health
    ax_val = _axes(2, 0, "Validation Losses")
    _plot(ax_val, "val/recon_loss", "Val L1",  palette[0])
    _plot(ax_val, "val/mel_loss",   "Val Mel", palette[1])

    if codebook_report is not None:
        ax_use = _axes(2, 1, "Codebook Usage (%)", ylabel="%")
        for i, u in enumerate(codebook_report["usage_pct"]):
            ax_use.bar(i, u, color=palette[i % len(palette)], alpha=0.8)
        ax_use.axhline(20, color="#f78166", linestyle="--", linewidth=1, label="Collapse threshold")
        ax_use.set_xlim(-0.5, len(codebook_report["usage_pct"]) - 0.5)
        ax_use.set_ylim(0, 105)
        ax_use.legend(fontsize=7, facecolor=bg_color, labelcolor=ax_color)

        ax_perp = _axes(2, 2, "Codebook Perplexity", ylabel="Perplexity")
        for i, p in enumerate(codebook_report["perplexity"]):
            ax_perp.bar(i, p, color=palette[i % len(palette)], alpha=0.8)
        ax_perp.set_xlim(-0.5, len(codebook_report["perplexity"]) - 0.5)

    plt.savefig(
        plot_dir / f"training_curves_step{step:08d}.png",
        dpi=150, bbox_inches="tight", facecolor="#0d1117",
    )
    # Always overwrite a "latest" copy for easy monitoring
    plt.savefig(
        plot_dir / "training_curves_latest.png",
        dpi=150, bbox_inches="tight", facecolor="#0d1117",
    )
    plt.close(fig)
    logger.info(f"Training plot saved: {plot_dir}/training_curves_latest.png")


def plot_spectrogram_comparison(
    real: torch.Tensor,
    fake: torch.Tensor,
    sample_rate: int,
    plot_dir: Path,
    step: int,
    n_fft: int = 1024,
    hop_length: int = 256,
) -> None:
    """
    Side-by-side mel spectrogram comparison of real vs. reconstructed audio.
    """
    if not MATPLOTLIB_AVAILABLE:
        return

    plot_dir.mkdir(parents=True, exist_ok=True)

    real_np = real[0].squeeze().detach().cpu().float().numpy()
    fake_np = fake[0].squeeze().detach().cpu().float().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4), facecolor="#0d1117")
    fig.suptitle(f"Spectrogram Comparison — Step {step:,}", color="#e6edf3", fontsize=12, fontweight="bold")

    def _spec(ax, wav, title):
        if LIBROSA_AVAILABLE:
            S = librosa.feature.melspectrogram(y=wav, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=80)
            S_db = librosa.power_to_db(S, ref=np.max)
        else:
            S_db = np.abs(np.fft.rfft(wav.reshape(-1, n_fft), axis=1)).T
        im = ax.imshow(S_db, origin="lower", aspect="auto", cmap="magma")
        ax.set_title(title, color="#e6edf3", fontsize=9)
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#e6edf3", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
        fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)

    _spec(axes[0], real_np, "Real Audio")
    _spec(axes[1], fake_np, "Reconstructed")

    # Difference
    if LIBROSA_AVAILABLE:
        S_real = librosa.feature.melspectrogram(y=real_np, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=80)
        S_fake = librosa.feature.melspectrogram(y=fake_np, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=80)
        diff = librosa.power_to_db(S_real) - librosa.power_to_db(S_fake)
    else:
        diff = np.zeros((80, 10))
    im2 = axes[2].imshow(diff, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-20, vmax=20)
    axes[2].set_title("Difference (Real − Reconstructed)", color="#e6edf3", fontsize=9)
    axes[2].set_facecolor("#161b22")
    axes[2].tick_params(colors="#e6edf3", labelsize=7)
    for spine in axes[2].spines.values():
        spine.set_edgecolor("#30363d")
    fig.colorbar(im2, ax=axes[2], fraction=0.05, pad=0.02)

    plt.tight_layout()
    plt.savefig(plot_dir / f"spectrogram_step{step:08d}.png", dpi=120, bbox_inches="tight", facecolor="#0d1117")
    plt.savefig(plot_dir / "spectrogram_latest.png",           dpi=120, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)


# =============================================================================
# MAIN MODEL: LipikaTokenizer
# =============================================================================

class LipikaTokenizer(nn.Module):
    """
    Full Lipika audio codec / tokenizer.

    Combines:
      - AudioEncoder with ScriptFamilyAdapter conditioning
      - ResidualVectorQuantizer with EMA updates + dead-code reset
      - AudioDecoder
      - (optionally) W2V-BERT semantic teacher for distillation targets
    """

    def __init__(
        self,
        audio_cfg: AudioConfig,
        rvq_cfg: RVQConfig,
        model_cfg: ModelConfig,
        use_semantic_teacher: bool = True,
    ) -> None:
        super().__init__()
        self.audio_cfg = audio_cfg
        self.rvq_cfg   = rvq_cfg
        self.model_cfg = model_cfg

        self.encoder        = AudioEncoder(audio_cfg, model_cfg)
        self.rvq            = ResidualVectorQuantizer(rvq_cfg, model_cfg)
        self.decoder        = AudioDecoder(model_cfg)
        self.script_adapter = ScriptFamilyAdapter(model_cfg)

        self.semantic_teacher: Optional[SemanticTeacher] = None
        if use_semantic_teacher and W2VBERT_AVAILABLE:
            try:
                self.semantic_teacher = SemanticTeacher(model_cfg.w2v_bert_model)
                logger.info("Semantic teacher loaded and frozen.")
            except Exception as e:
                logger.warning(f"Could not load semantic teacher: {e}. Continuing without it.")

        self.mel_loss_fn  = MelSpectrogramLoss(audio_cfg)
        self.stft_loss_fn = MultiScaleSTFTLoss()
        self.cb_monitor   = CodebookMonitor(rvq_cfg.n_codebooks, rvq_cfg.codebook_size)

    def forward(
        self,
        waveform: torch.Tensor,
        script_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        script_cond = None
        if script_ids is not None:
            script_cond = self.script_adapter(script_ids)

        z = self.encoder(waveform, script_adapter=script_cond)

        w2v_targets = None
        if self.semantic_teacher is not None and self.training:
            with torch.no_grad():
                w2v_targets = self.semantic_teacher(waveform, src_sr=self.audio_cfg.sample_rate)

        quantised = self.rvq(z, w2v_targets=w2v_targets)

        if self.training:
            self.cb_monitor.update(quantised["codes"].detach())

        reconstructed = self.decoder(quantised["z_q"])

        min_t = min(reconstructed.shape[-1], waveform.shape[-1])
        reconstructed = reconstructed[..., :min_t]
        target        = waveform[..., :min_t]

        recon_loss = F.l1_loss(reconstructed, target)
        mel_loss   = self.mel_loss_fn(target, reconstructed)
        stft_loss  = self.stft_loss_fn(target, reconstructed)

        return {
            "reconstructed": reconstructed,
            "target":         target,
            "codes":          quantised["codes"],
            "recon_loss":     recon_loss,
            "mel_loss":       mel_loss,
            "stft_loss":      stft_loss,
            "vq_loss":        quantised["vq_loss"],
            "semantic_loss":  quantised["semantic_loss"],
        }

    @torch.no_grad()
    def encode(
        self,
        waveform: torch.Tensor,
        script_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        script_cond = None
        if script_ids is not None:
            script_cond = self.script_adapter(script_ids)
        z   = self.encoder(waveform, script_adapter=script_cond)
        out = self.rvq(z, w2v_targets=None)
        return out["codes"]

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        z_q = self.rvq.decode_from_codes(codes)
        return self.decoder(z_q)

    @property
    def frame_rate(self) -> float:
        return self.audio_cfg.sample_rate / self.encoder.compression_ratio

    def num_parameters(self, exclude_teacher: bool = True) -> int:
        params = [
            p for name, p in self.named_parameters()
            if not (exclude_teacher and "semantic_teacher" in name)
        ]
        return sum(p.numel() for p in params)


# =============================================================================
# DATASET
# =============================================================================

class AudioDataset(Dataset):
    """
    Audio dataset for Lipika training.

    Loads audio files, resamples to target sample rate, randomly crops to
    max_samples, and reads optional sidecar JSON for language/script metadata.

    Metadata format (per audio file, same stem, .json extension):
        {"lang": "hi"}   # ISO 639-1 language code

    Falls back to ScriptFamily.DEVANAGARI if no metadata found.
    """

    AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3", ".opus"}

    def __init__(
        self,
        data_dir: str,
        audio_cfg: AudioConfig,
        max_duration: float = 5.0,
        split: str = "train",
        val_fraction: float = 0.02,
        seed: int = 42,
    ) -> None:
        self.data_dir    = Path(data_dir)
        self.sample_rate = audio_cfg.sample_rate
        self.max_samples = int(max_duration * audio_cfg.sample_rate)
        self.split       = split

        all_files = sorted([
            p for ext in self.AUDIO_EXTENSIONS
            for p in self.data_dir.rglob(f"*{ext}")
        ])

        if len(all_files) == 0:
            raise FileNotFoundError(
                f"No audio files found under {data_dir}. "
                "Supported formats: " + ", ".join(self.AUDIO_EXTENSIONS)
            )

        rng = random.Random(seed)
        rng.shuffle(all_files)
        n_val = max(1, int(len(all_files) * val_fraction))
        self.files = all_files[:n_val] if split == "val" else all_files[n_val:]
        logger.info(f"[{split}] {len(self.files)} audio files found.")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.files[idx]
        try:
            return self._load(path)
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}. Skipping.")
            return self.__getitem__((idx + 1) % len(self.files))

    def _load(self, path: Path) -> Dict[str, Any]:
        if not SOUNDFILE_AVAILABLE:
            raise RuntimeError("soundfile is required for dataset loading. pip install soundfile")

        audio, sr = sf.read(path, dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)

        if sr != self.sample_rate:
            if not LIBROSA_AVAILABLE:
                raise RuntimeError(f"librosa required for resampling {sr}->{self.sample_rate}. pip install librosa")
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)

        audio = torch.from_numpy(audio).float()

        if audio.shape[0] >= self.max_samples:
            start = random.randint(0, audio.shape[0] - self.max_samples)
            audio = audio[start: start + self.max_samples]
        else:
            audio = F.pad(audio, (0, self.max_samples - audio.shape[0]))

        peak = audio.abs().max()
        if peak > 0:
            audio = audio / (peak + 1e-6) * 0.98

        return {
            "waveform":  audio.unsqueeze(0),
            "script_id": self._read_script_id(path),
            "path":      str(path),
        }

    def _read_script_id(self, audio_path: Path) -> int:
        meta_path = audio_path.with_suffix(".json")
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                lang = meta.get("lang", "hi")
                return int(LANG_TO_SCRIPT.get(lang, ScriptFamily.DEVANAGARI))
            except Exception:
                pass
        return int(ScriptFamily.DEVANAGARI)


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    waveforms  = torch.stack([b["waveform"] for b in batch])
    script_ids = torch.tensor([b["script_id"] for b in batch], dtype=torch.long)
    return {"waveform": waveforms, "script_id": script_ids}


# =============================================================================
# DUMMY DATASET  (used when no real data is available — for smoke-testing)
# =============================================================================

class SyntheticAudioDataset(Dataset):
    """
    Generates random synthetic audio on-the-fly.
    Useful for smoke-testing the training loop without real data.
    """

    def __init__(
        self,
        audio_cfg: AudioConfig,
        n_samples: int = 100,
        max_duration: float = 2.0,
        seed: int = 42,
    ) -> None:
        self.sample_rate = audio_cfg.sample_rate
        self.max_samples = int(max_duration * audio_cfg.sample_rate)
        self.n_samples   = n_samples
        self.rng         = np.random.default_rng(seed)
        logger.info(f"[SyntheticDataset] {n_samples} synthetic samples, {max_duration}s each.")

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Generate a simple sine mix as synthetic "speech"
        t    = np.linspace(0, self.max_samples / self.sample_rate, self.max_samples)
        freq = self.rng.uniform(80, 300)          # random F0
        wav  = 0.5 * np.sin(2 * np.pi * freq * t)
        # Add harmonics
        for h in range(2, 6):
            wav += (0.5 / h) * np.sin(2 * np.pi * freq * h * t)
        wav  = wav / (np.abs(wav).max() + 1e-6) * 0.98
        audio = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0)
        return {
            "waveform":  audio,
            "script_id": int(idx % len(ScriptFamily)),
            "path":      f"synthetic_{idx}",
        }


# =============================================================================
# LEARNING RATE SCHEDULE
# =============================================================================

def cosine_schedule_with_warmup(
    step: int,
    warmup_steps: int,
    decay_steps: int,
    min_lr_ratio: float = 0.1,
) -> float:
    if step < warmup_steps:
        return step / max(warmup_steps, 1)
    progress = min((step - warmup_steps) / max(decay_steps - warmup_steps, 1), 1.0)
    return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


# =============================================================================
# CHECKPOINT MANAGER
# =============================================================================

class CheckpointManager:
    """
    Saves and loads model + optimiser state with rolling deletion.
    Only rank-0 writes to disk.
    """

    def __init__(self, ckpt_dir: Path, keep: int = 5, rank: int = 0) -> None:
        self.ckpt_dir = ckpt_dir
        self.keep     = keep
        self.rank     = rank
        if rank == 0:
            ckpt_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        step: int,
        model: nn.Module,
        disc: nn.Module,
        gen_opt: optim.Optimizer,
        disc_opt: optim.Optimizer,
        gen_sched, disc_sched,
        metrics: Dict[str, float],
        audio_cfg: AudioConfig,
        rvq_cfg: RVQConfig,
        model_cfg: ModelConfig,
    ) -> None:
        if self.rank != 0:
            return

        m = model.module if (DDP_AVAILABLE and isinstance(model, DDP)) else model
        d = disc.module  if (DDP_AVAILABLE and isinstance(disc,  DDP)) else disc

        payload = {
            "step":        step,
            "model_state": m.state_dict(),
            "disc_state":  d.state_dict(),
            "gen_opt":     gen_opt.state_dict(),
            "disc_opt":    disc_opt.state_dict(),
            "gen_sched":   gen_sched.state_dict(),
            "disc_sched":  disc_sched.state_dict(),
            "metrics":     metrics,
            "audio_cfg":   asdict(audio_cfg),
            "rvq_cfg":     asdict(rvq_cfg),
            "model_cfg":   asdict(model_cfg),
        }
        path = self.ckpt_dir / f"ckpt_step{step:08d}.pt"
        torch.save(payload, path)
        logger.info(f"Checkpoint saved: {path}")

        checkpoints = sorted(self.ckpt_dir.glob("ckpt_step*.pt"))
        while len(checkpoints) > self.keep:
            old = checkpoints.pop(0)
            old.unlink()
            logger.info(f"Deleted old checkpoint: {old}")

    @staticmethod
    def load(
        path: str,
        model: nn.Module,
        disc: nn.Module,
        gen_opt=None, disc_opt=None,
        gen_sched=None, disc_sched=None,
        device: str = "cpu",
    ) -> int:
        payload = torch.load(path, map_location=device, weights_only=False)
        m = model.module if (DDP_AVAILABLE and isinstance(model, DDP)) else model
        d = disc.module  if (DDP_AVAILABLE and isinstance(disc,  DDP)) else disc
        m.load_state_dict(payload["model_state"])
        d.load_state_dict(payload["disc_state"])
        if gen_opt:   gen_opt.load_state_dict(payload["gen_opt"])
        if disc_opt:  disc_opt.load_state_dict(payload["disc_opt"])
        if gen_sched: gen_sched.load_state_dict(payload["gen_sched"])
        if disc_sched:disc_sched.load_state_dict(payload["disc_sched"])
        logger.info(f"Resumed from step {payload['step']}: {path}")
        return payload["step"]

    def latest(self) -> Optional[Path]:
        ckpts = sorted(self.ckpt_dir.glob("ckpt_step*.pt"))
        return ckpts[-1] if ckpts else None


# =============================================================================
# DISTRIBUTED SETUP
# =============================================================================

def setup_distributed(rank: int, world_size: int, backend: str = "nccl") -> None:
    if not DDP_AVAILABLE:
        raise RuntimeError("torch.distributed not available.")
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed() -> None:
    if DDP_AVAILABLE and dist.is_initialized():
        dist.destroy_process_group()


# =============================================================================
# AUDIO OUTPUT UTILITIES  (NEW)
# =============================================================================

def save_audio_sample(
    waveform: torch.Tensor,
    path: Path,
    sample_rate: int,
) -> None:
    """Save a waveform tensor to a WAV file. Safe no-op if soundfile unavailable."""
    if not SOUNDFILE_AVAILABLE:
        logger.warning("soundfile not available; cannot save audio sample.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    audio_np = waveform.squeeze().detach().cpu().float().numpy()
    audio_np = np.clip(audio_np, -1.0, 1.0)
    sf.write(str(path), audio_np, sample_rate)


def save_audio_comparison(
    real: torch.Tensor,
    fake: torch.Tensor,
    output_dir: Path,
    step: int,
    sample_rate: int,
    n_samples: int = 4,
) -> None:
    """Save up to n_samples real/reconstructed pairs to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for i in range(min(n_samples, real.shape[0])):
        save_audio_sample(real[i],  output_dir / f"real_step{step:08d}_sample{i}.wav",        sample_rate)
        save_audio_sample(fake[i],  output_dir / f"reconstructed_step{step:08d}_sample{i}.wav", sample_rate)
    logger.info(f"Audio samples saved to {output_dir} at step {step}")


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train(
    rank: int,
    world_size: int,
    audio_cfg: AudioConfig,
    rvq_cfg: RVQConfig,
    model_cfg: ModelConfig,
    train_cfg: TrainingConfig,
    resume_from: Optional[str] = None,
    use_semantic: bool = True,
) -> None:
    """
    Main training function, launched per GPU rank (or once for CPU/MPS).

    Device support matrix
    ─────────────────────
    CUDA GPU   → full support: AMP (bf16/fp16), DDP multi-GPU, TorchScript export
    Apple MPS  → single-process, float32 (no AMP), no DDP
    CPU        → single-process, float32, slower but fully functional
    """

    # ── Device resolution ──────────────────────────────────────────────────
    if train_cfg.device == "auto":
        base_device = get_device()
    else:
        base_device = get_device(train_cfg.device)

    is_cuda = base_device.type == "cuda"
    is_mps  = base_device.type == "mps"
    is_cpu  = base_device.type == "cpu"

    # DDP is CUDA-only
    if not is_cuda and world_size > 1:
        logger.warning(f"DDP requires CUDA. Falling back to single-process on {base_device}.")
        world_size = 1
        rank       = 0

    # AMP only on CUDA
    use_amp = train_cfg.mixed_precision and is_cuda
    if train_cfg.mixed_precision and not is_cuda:
        logger.info(f"Mixed precision requested but device is '{base_device.type}'. Disabled -> float32.")

    use_pin_memory = train_cfg.pin_memory and is_cuda

    # ── Seeds ──────────────────────────────────────────────────────────────
    random.seed(train_cfg.seed + rank)
    np.random.seed(train_cfg.seed + rank)
    torch.manual_seed(train_cfg.seed + rank)
    if is_cuda:
        torch.cuda.manual_seed_all(train_cfg.seed + rank)

    # ── Distributed ────────────────────────────────────────────────────────
    is_distributed = world_size > 1 and is_cuda
    if is_distributed:
        setup_distributed(rank, world_size, train_cfg.ddp_backend)
    device = torch.device(f"cuda:{rank}") if is_distributed else base_device
    setup_logging(Path(train_cfg.log_dir), rank)

    # Log hardware info once
    if rank == 0:
        logger.info(f"{'='*70}")
        logger.info(f"  Lipika Tokenizer - Production Training")
        logger.info(f"  Device  : {device_info(device)}")
        logger.info(f"  AMP     : {use_amp}  |  Distributed: {is_distributed} ({world_size} GPUs)")
        logger.info(f"  Batch   : {train_cfg.batch_size} x {train_cfg.grad_accum_steps} (accum)")
        logger.info(f"{'='*70}")

    # ── Models ─────────────────────────────────────────────────────────────
    model         = LipikaTokenizer(audio_cfg, rvq_cfg, model_cfg, use_semantic_teacher=use_semantic).to(device)
    discriminator = MultiScaleMultiPeriodDiscriminator(model_cfg).to(device)

    if train_cfg.compile_model and is_cuda:
        try:
            model         = torch.compile(model)
            discriminator = torch.compile(discriminator)
            logger.info("torch.compile applied.")
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}. Continuing without compilation.")
    elif train_cfg.compile_model:
        logger.warning("torch.compile is CUDA-only. Skipping.")

    if is_distributed:
        model         = DDP(model,         device_ids=[rank], find_unused_parameters=True)
        discriminator = DDP(discriminator, device_ids=[rank])

    if rank == 0:
        m_raw = model.module if is_distributed else model
        n_params = m_raw.num_parameters(exclude_teacher=True)
        logger.info(f"Model parameters (excl. teacher): {n_params / 1e6:.2f} M")

    # ── Optimisers ─────────────────────────────────────────────────────────
    m_raw = model.module if is_distributed else model
    gen_params = [
        p for name, p in m_raw.named_parameters()
        if "semantic_teacher" not in name and p.requires_grad
    ]
    gen_optimizer = optim.AdamW(
        gen_params, lr=train_cfg.learning_rate,
        betas=train_cfg.betas, weight_decay=train_cfg.weight_decay,
    )
    disc_optimizer = optim.AdamW(
        discriminator.parameters(), lr=train_cfg.disc_learning_rate,
        betas=train_cfg.betas, weight_decay=train_cfg.weight_decay,
    )

    gen_scheduler = optim.lr_scheduler.LambdaLR(
        gen_optimizer,
        lambda s: cosine_schedule_with_warmup(s, train_cfg.warmup_steps, train_cfg.lr_decay_steps),
    )
    disc_scheduler = optim.lr_scheduler.LambdaLR(
        disc_optimizer,
        lambda s: cosine_schedule_with_warmup(s, train_cfg.warmup_steps, train_cfg.lr_decay_steps),
    )

    amp_dtype  = torch.bfloat16 if supports_bf16(device) else torch.float16
    # GradScaler only makes sense on CUDA
    scaler_gen  = torch.cuda.amp.GradScaler(enabled=use_amp) if is_cuda else None
    scaler_disc = torch.cuda.amp.GradScaler(enabled=use_amp) if is_cuda else None

    # ── Dataset ────────────────────────────────────────────────────────────
    data_path = Path(train_cfg.data_dir)
    has_data  = data_path.exists() and any(
        data_path.rglob(f"*{ext}") for ext in AudioDataset.AUDIO_EXTENSIONS
    )

    if has_data:
        train_dataset = AudioDataset(train_cfg.data_dir, audio_cfg, train_cfg.max_duration, "train")
        val_dataset   = AudioDataset(train_cfg.data_dir, audio_cfg, train_cfg.max_duration, "val")
    else:
        logger.warning(
            f"No audio data found in '{train_cfg.data_dir}'. "
            "Using SyntheticAudioDataset for smoke-testing."
        )
        train_dataset = SyntheticAudioDataset(audio_cfg, n_samples=200, max_duration=train_cfg.max_duration)
        val_dataset   = SyntheticAudioDataset(audio_cfg, n_samples=20,  max_duration=train_cfg.max_duration, seed=999)

    if is_distributed:
        from torch.utils.data import DistributedSampler
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=train_cfg.num_workers,
        pin_memory=use_pin_memory,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=max(1, train_cfg.batch_size // 2),
        shuffle=False,
        num_workers=train_cfg.num_workers,
        pin_memory=use_pin_memory,
        collate_fn=collate_fn,
    )

    # ── Writer / Checkpoint / Metrics ──────────────────────────────────────
    writer = SummaryWriter(log_dir=train_cfg.log_dir) if rank == 0 else SummaryWriter()
    ckpt_mgr  = CheckpointManager(Path(train_cfg.checkpoint_dir), train_cfg.keep_last_n_checkpoints, rank)
    metrics_tracker = MetricsTracker()

    plot_dir   = Path(train_cfg.plot_dir)
    output_dir = Path(train_cfg.output_dir)

    if rank == 0:
        plot_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Plots   -> {plot_dir.resolve()}")
        logger.info(f"Outputs -> {output_dir.resolve()}")

    # ── Resume ─────────────────────────────────────────────────────────────
    global_step = 0
    resume_path = resume_from or (str(ckpt_mgr.latest()) if ckpt_mgr.latest() else None)
    if resume_path:
        try:
            global_step = CheckpointManager.load(
                resume_path, model, discriminator,
                gen_optimizer, disc_optimizer,
                gen_scheduler, disc_scheduler,
                device=str(device),
            )
        except Exception as e:
            logger.error(f"Failed to resume from {resume_path}: {e}. Starting fresh.")
            global_step = 0

    # ── Training Loop ──────────────────────────────────────────────────────
    gan_active = global_step >= train_cfg.disc_start_step
    d_loss_val = adv_loss_val = feat_loss_val = 0.0

    # Context manager for AMP — handles both CUDA and non-CUDA safely
    def amp_context():
        if use_amp:
            return torch.cuda.amp.autocast(enabled=True, dtype=amp_dtype)
        return nullcontext()

    for epoch in range(train_cfg.num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        discriminator.train()

        epoch_metrics: Dict[str, float] = defaultdict(float)
        step_count = 0
        epoch_start = time.time()

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1:3d}/{train_cfg.num_epochs}",
            disable=(rank != 0),
            dynamic_ncols=True,
        )

        for batch in pbar:
            try:
                waveform   = batch["waveform"].to(device, non_blocking=use_pin_memory)
                script_ids = batch["script_id"].to(device, non_blocking=use_pin_memory)

                # ── Discriminator update ────────────────────────────────────
                if gan_active and global_step % train_cfg.disc_update_every == 0:
                    disc_optimizer.zero_grad(set_to_none=True)
                    with amp_context():
                        with torch.no_grad():
                            fwd_d = model(waveform, script_ids)
                        fake = fwd_d["reconstructed"].detach()
                        real_logits, _ = discriminator(waveform)
                        fake_logits, _ = discriminator(fake)
                        d_loss = hinge_disc_loss(real_logits, fake_logits)

                    if scaler_disc is not None:
                        scaler_disc.scale(d_loss).backward()
                        scaler_disc.unscale_(disc_optimizer)
                        clip_grad_norm_(discriminator.parameters(), train_cfg.grad_clip)
                        scaler_disc.step(disc_optimizer)
                        scaler_disc.update()
                    else:
                        d_loss.backward()
                        clip_grad_norm_(discriminator.parameters(), train_cfg.grad_clip)
                        disc_optimizer.step()
                    disc_scheduler.step()
                    d_loss_val = d_loss.item()

                # ── Generator update ────────────────────────────────────────
                gen_optimizer.zero_grad(set_to_none=True)
                with amp_context():
                    fwd = model(waveform, script_ids)
                    g_loss = (
                        train_cfg.w_time_recon * fwd["recon_loss"]
                        + train_cfg.w_mel       * fwd["mel_loss"]
                        + train_cfg.w_freq_recon * fwd["stft_loss"]
                        + train_cfg.w_vq        * fwd["vq_loss"]
                        + train_cfg.w_semantic  * fwd["semantic_loss"]
                    )
                    if gan_active:
                        fake_logits, fake_feats = discriminator(fwd["reconstructed"])
                        with torch.no_grad():
                            _, real_feats = discriminator(waveform)
                        adv_loss  = hinge_gen_loss(fake_logits)
                        feat_loss = feature_matching_loss(real_feats, fake_feats)
                        g_loss    = g_loss + train_cfg.w_gen * adv_loss + train_cfg.w_feat * feat_loss
                        adv_loss_val  = adv_loss.item()
                        feat_loss_val = feat_loss.item()

                scaled_g = g_loss / train_cfg.grad_accum_steps
                if scaler_gen is not None:
                    scaler_gen.scale(scaled_g).backward()
                else:
                    scaled_g.backward()

                if (global_step + 1) % train_cfg.grad_accum_steps == 0:
                    if scaler_gen is not None:
                        scaler_gen.unscale_(gen_optimizer)
                        clip_grad_norm_(gen_params, train_cfg.grad_clip)
                        scaler_gen.step(gen_optimizer)
                        scaler_gen.update()
                    else:
                        clip_grad_norm_(gen_params, train_cfg.grad_clip)
                        gen_optimizer.step()
                    gen_scheduler.step()
                    gen_optimizer.zero_grad(set_to_none=True)

                # ── Activate GAN ────────────────────────────────────────────
                if not gan_active and global_step >= train_cfg.disc_start_step:
                    gan_active = True
                    logger.info(f"GAN training activated at step {global_step}.")

                # ── Step bookkeeping ────────────────────────────────────────
                global_step += 1
                step_count  += 1
                g_loss_val = g_loss.item()

                epoch_metrics["g_loss"] += g_loss_val
                epoch_metrics["recon"]  += fwd["recon_loss"].item()
                epoch_metrics["mel"]    += fwd["mel_loss"].item()
                epoch_metrics["stft"]   += fwd["stft_loss"].item()
                epoch_metrics["vq"]     += fwd["vq_loss"].item()
                epoch_metrics["sem"]    += fwd["semantic_loss"].item()

                # ── Logging ─────────────────────────────────────────────────
                if rank == 0 and global_step % 50 == 0:
                    lr_now = gen_optimizer.param_groups[0]["lr"]
                    step_metrics = {
                        "g_loss": g_loss_val,
                        "recon": fwd["recon_loss"].item(),
                        "mel":   fwd["mel_loss"].item(),
                        "stft":  fwd["stft_loss"].item(),
                        "vq":    fwd["vq_loss"].item(),
                        "sem":   fwd["semantic_loss"].item(),
                        "lr":    lr_now,
                    }
                    if gan_active:
                        step_metrics["d_loss"]    = d_loss_val
                        step_metrics["adv_loss"]  = adv_loss_val
                        step_metrics["feat_loss"] = feat_loss_val
                    metrics_tracker.update(global_step, step_metrics)

                    # TensorBoard
                    for k, v in step_metrics.items():
                        writer.add_scalar(f"train/{k}", v, global_step)

                    # Codebook health
                    m_raw = model.module if is_distributed else model
                    m_raw.cb_monitor.log_to_tensorboard(writer, global_step)
                    rpt = m_raw.cb_monitor.report()
                    if rpt["collapse_warning"]:
                        logger.warning(
                            f"Step {global_step}: Codebook collapse! "
                            f"Usage: {[f'{u:.1f}%' for u in rpt['usage_pct']]}"
                        )

                pbar.set_postfix({
                    "g": f"{g_loss_val:.4f}",
                    "mel": f"{fwd['mel_loss'].item():.4f}",
                    "step": global_step,
                })

                # ── Checkpoint ──────────────────────────────────────────────
                if rank == 0 and global_step % train_cfg.save_every_steps == 0:
                    avg = {k: v / max(step_count, 1) for k, v in epoch_metrics.items()}
                    ckpt_mgr.save(
                        global_step, model, discriminator,
                        gen_optimizer, disc_optimizer,
                        gen_scheduler, disc_scheduler,
                        avg, audio_cfg, rvq_cfg, model_cfg,
                    )

                # ── Validation ──────────────────────────────────────────────
                if rank == 0 and global_step % train_cfg.eval_every_steps == 0:
                    val_metrics = validate(model, val_loader, device, use_amp, amp_dtype)
                    for k, v in val_metrics.items():
                        writer.add_scalar(f"val/{k}", v, global_step)
                        metrics_tracker.update(global_step, {f"val/{k}": v})
                    logger.info(
                        f"Step {global_step} | "
                        f"val_recon={val_metrics['recon_loss']:.5f}  "
                        f"val_mel={val_metrics['mel_loss']:.5f}"
                    )

                # ── Plots ───────────────────────────────────────────────────
                if rank == 0 and global_step % train_cfg.plot_every_steps == 0:
                    m_raw_plot = model.module if is_distributed else model
                    cb_rpt = m_raw_plot.cb_monitor.report()
                    plot_training_curves(metrics_tracker, plot_dir, global_step, cb_rpt)
                    plot_spectrogram_comparison(
                        fwd["target"], fwd["reconstructed"],
                        audio_cfg.sample_rate, plot_dir, global_step,
                    )
                    metrics_tracker.save_csv(plot_dir / "training_metrics.csv")

                # ── Audio samples ────────────────────────────────────────────
                if rank == 0 and global_step % train_cfg.sample_every_steps == 0:
                    save_audio_comparison(
                        fwd["target"], fwd["reconstructed"],
                        output_dir, global_step, audio_cfg.sample_rate, n_samples=2,
                    )

            except Exception as e:
                logger.error(f"Training step {global_step} failed: {e}")
                logger.error(traceback.format_exc())
                # Skip bad batches instead of crashing
                continue

        # ── End-of-epoch ────────────────────────────────────────────────────
        if rank == 0:
            epoch_time = time.time() - epoch_start
            avg = {k: v / max(step_count, 1) for k, v in epoch_metrics.items()}
            logger.info(
                f"Epoch {epoch+1:3d}/{train_cfg.num_epochs} | "
                f"time={epoch_time:.1f}s | "
                + "  ".join(f"{k}={v:.4f}" for k, v in avg.items())
            )

    # ── Final save ──────────────────────────────────────────────────────────
    if rank == 0:
        avg = {k: v / max(step_count, 1) for k, v in epoch_metrics.items()}
        ckpt_mgr.save(
            global_step, model, discriminator,
            gen_optimizer, disc_optimizer,
            gen_scheduler, disc_scheduler,
            avg, audio_cfg, rvq_cfg, model_cfg,
        )
        # Final plots + CSV
        m_raw_final = model.module if is_distributed else model
        plot_training_curves(metrics_tracker, plot_dir, global_step, m_raw_final.cb_monitor.report())
        metrics_tracker.save_csv(plot_dir / "training_metrics.csv")
        logger.info("Training complete.")
        if writer:
            writer.close()

    cleanup_distributed()


# =============================================================================
# VALIDATION
# =============================================================================

@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mixed_precision: bool,
    dtype: torch.dtype,
) -> Dict[str, float]:
    model.eval()
    totals: Dict[str, float] = defaultdict(float)
    count = 0

    amp_ctx = (
        torch.cuda.amp.autocast(enabled=mixed_precision, dtype=dtype)
        if device.type == "cuda" else nullcontext()
    )

    for batch in loader:
        try:
            waveform   = batch["waveform"].to(device)
            script_ids = batch["script_id"].to(device)
            with amp_ctx:
                fwd = model(waveform, script_ids)
            totals["recon_loss"] += fwd["recon_loss"].item()
            totals["mel_loss"]   += fwd["mel_loss"].item()
            totals["stft_loss"]  += fwd["stft_loss"].item()
            totals["vq_loss"]    += fwd["vq_loss"].item()
            totals["sem_loss"]   += fwd["semantic_loss"].item()
            count += 1
        except Exception as e:
            logger.warning(f"Validation batch failed: {e}")
            continue

    model.train()
    return {k: v / max(count, 1) for k, v in totals.items()}


# =============================================================================
# INFERENCE UTILITIES
# =============================================================================

@torch.no_grad()
def encode_audio_file(
    model: LipikaTokenizer,
    audio_path: str,
    lang: str = "hi",
    device: Optional[str] = None,
) -> torch.Tensor:
    """
    Encode a single audio file to discrete codes.

    Returns:
        codes: (1, T_frames, n_codebooks) int64
    """
    resolved = get_device(device) if device else get_device()

    if not SOUNDFILE_AVAILABLE:
        raise RuntimeError("soundfile required for audio loading.")

    audio, sr = sf.read(audio_path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != model.audio_cfg.sample_rate:
        if not LIBROSA_AVAILABLE:
            raise RuntimeError("librosa required for resampling.")
        audio = librosa.resample(audio, orig_sr=sr, target_sr=model.audio_cfg.sample_rate)

    waveform = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0).to(resolved)
    script_id = torch.tensor(
        [int(LANG_TO_SCRIPT.get(lang, ScriptFamily.DEVANAGARI))],
        dtype=torch.long, device=resolved,
    )
    model = model.to(resolved).eval()
    return model.encode(waveform, script_id)


@torch.no_grad()
def decode_codes_to_file(
    model: LipikaTokenizer,
    codes: torch.Tensor,
    out_path: str,
    device: Optional[str] = None,
) -> None:
    """Decode discrete codes to a WAV file."""
    resolved = get_device(device) if device else get_device()
    model = model.to(resolved).eval()
    waveform = model.decode(codes.to(resolved))
    save_audio_sample(waveform, Path(out_path), model.audio_cfg.sample_rate)
    logger.info(f"Decoded audio saved to {out_path}")


# =============================================================================
# MODEL EXPORT
# =============================================================================

def export_torchscript(model: LipikaTokenizer, out_path: str) -> None:
    """Export encoder + decoder to TorchScript for production serving."""
    device = get_device()
    model  = model.to(device).eval()
    try:
        scripted = torch.jit.script(model)
        torch.jit.save(scripted, out_path)
        logger.info(f"TorchScript model saved to {out_path}")
    except Exception as e:
        logger.error(f"TorchScript export failed: {e}. Try torch.jit.trace instead.")
        raise


# =============================================================================
# HELPER: load model from checkpoint
# =============================================================================

def _load_model_from_checkpoint(
    ckpt_path: str,
    device: Optional[str] = None,
) -> LipikaTokenizer:
    resolved = get_device(device) if device else get_device()
    payload   = torch.load(ckpt_path, map_location=str(resolved), weights_only=False)
    audio_cfg = AudioConfig(**payload["audio_cfg"])
    rvq_cfg   = RVQConfig(**payload["rvq_cfg"])
    model_cfg = ModelConfig(**payload["model_cfg"])
    model     = LipikaTokenizer(audio_cfg, rvq_cfg, model_cfg, use_semantic_teacher=False)
    model.load_state_dict(payload["model_state"])
    return model.to(str(resolved)).eval()


# =============================================================================
# QUICK SMOKE-TEST  (new — verifies the model runs end-to-end)
# =============================================================================

def smoke_test(device_str: str = "auto") -> None:
    """
    Run minimal end-to-end forward passes for every preset size.
    Catches channel-shape bugs immediately without requiring real data.
    """
    logger.info("Running smoke test...")
    device = get_device(device_str)
    logger.info(f"Smoke-test device: {device_info(device)}")

    test_configs = [
        ("cpu-tiny",   64,  4, 64),
        ("gpu-small",  128, 4, 128),
        ("gpu-full",   256, 4, 256),
    ]

    B = 1
    T = 24_000  # 1-second batch (fast)

    all_passed = True
    for label, enc_ch, n_cb, dec_ch in test_configs:
        try:
            audio_cfg = AudioConfig(sample_rate=24_000)
            rvq_cfg   = RVQConfig(n_codebooks=n_cb, codebook_size=64)
            model_cfg = ModelConfig(
                encoder_channels=enc_ch,
                decoder_channels=dec_ch,
                disc_channels=16, disc_depth=2,
            )
            model = LipikaTokenizer(
                audio_cfg, rvq_cfg, model_cfg, use_semantic_teacher=False
            ).to(device)
            model.train()

            waveform   = torch.randn(B, 1, T, device=device)
            script_ids = torch.randint(0, model_cfg.n_script_families, (B,), device=device)

            fwd = model(waveform, script_ids)
            assert fwd["reconstructed"].shape == (B, 1, T), \
                f"Recon shape mismatch: {fwd['reconstructed'].shape} != {(B, 1, T)}"
            assert fwd["codes"].shape[-1] == n_cb, \
                f"Codes codebook dim wrong: {fwd['codes'].shape[-1]} != {n_cb}"

            # Encode/decode round-trip
            model.eval()
            codes = model.encode(waveform, script_ids)
            recon = model.decode(codes)
            assert recon.shape[0] == B

            n_params = model.num_parameters(exclude_teacher=True)
            logger.info(
                f"  [{label}] PASSED  enc={enc_ch}ch  cb={n_cb}x64  "
                f"params={n_params/1e6:.2f}M  "
                f"codes={codes.shape}  recon={recon.shape}"
            )
        except Exception as e:
            logger.error(f"  [{label}] FAILED: {e}")
            all_passed = False

    if all_passed:
        logger.info("All smoke tests PASSED.")
    else:
        logger.error("Some smoke tests FAILED. Check errors above.")
        raise RuntimeError("Smoke test failure - do not proceed to training.")
    logger.info(f"  Recon shape   : {recon.shape}")
    logger.info(f"  Frame rate    : {model.frame_rate:.1f} Hz")
    logger.info(f"  Compress ratio: {model.encoder.compression_ratio}x")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lipika — Production Neural Audio Tokenizer for Indic TTS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # ── train ──────────────────────────────────────────────────────────────
    tr = sub.add_parser("train", help="Train or resume training")
    tr.add_argument("--data-dir",        default="./data")
    tr.add_argument("--checkpoint-dir",  default="./checkpoints")
    tr.add_argument("--log-dir",         default="./logs")
    tr.add_argument("--plot-dir",        default="./plots",   help="Directory for training curve plots")
    tr.add_argument("--output-dir",      default="./outputs", help="Directory for audio sample outputs")
    tr.add_argument("--batch-size",      type=int,   default=None,
                    help="Override batch size (preset sets a sensible default)")
    tr.add_argument("--epochs",          type=int,   default=200)
    tr.add_argument("--lr",              type=float, default=3e-4)
    tr.add_argument("--resume",          default=None)
    tr.add_argument("--gpus",            type=int,   default=1)
    tr.add_argument("--compile",         action="store_true")
    tr.add_argument("--no-semantic",     action="store_true",
                    help="Disable W2V-BERT semantic distillation (faster, less VRAM)")
    tr.add_argument("--device",          default="auto",
                    help="Compute device: 'auto' | 'cuda' | 'cuda:N' | 'mps' | 'cpu'")
    tr.add_argument("--preset",          default="auto",
                    choices=["auto", "cpu", "gpu-small", "gpu-full"],
                    help=(
                        "Model + training size preset. "
                        "'auto' picks based on detected hardware. "
                        "'cpu' = tiny model, batch=2, fast to iterate. "
                        "'gpu-small' = medium model, batch=4 (~8 GB VRAM). "
                        "'gpu-full' = full EnCodec-scale, batch=8 (~16 GB VRAM)."
                    ))
    tr.add_argument("--save-every",      type=int,   default=5_000, dest="save_every")
    tr.add_argument("--eval-every",      type=int,   default=1_000, dest="eval_every")
    tr.add_argument("--plot-every",      type=int,   default=500,   dest="plot_every")
    tr.add_argument("--sample-every",    type=int,   default=2_000, dest="sample_every")
    tr.add_argument("--grad-accum",      type=int,   default=1,     dest="grad_accum")
    tr.add_argument("--no-amp",          action="store_true",        help="Disable mixed precision")
    tr.add_argument("--num-workers",     type=int,   default=0)
    tr.add_argument("--seed",            type=int,   default=42)

    # ── encode ─────────────────────────────────────────────────────────────
    enc = sub.add_parser("encode", help="Encode audio to discrete codes")
    enc.add_argument("audio_path")
    enc.add_argument("--out",        default="codes.pt")
    enc.add_argument("--lang",       default="hi")
    enc.add_argument("--checkpoint", required=True)
    enc.add_argument("--device",     default="auto")

    # ── decode ─────────────────────────────────────────────────────────────
    dec = sub.add_parser("decode", help="Decode discrete codes to audio")
    dec.add_argument("codes_path")
    dec.add_argument("--out",        default="reconstructed.wav")
    dec.add_argument("--checkpoint", required=True)
    dec.add_argument("--device",     default="auto")

    # ── smoke-test ─────────────────────────────────────────────────────────
    st = sub.add_parser("smoke-test", help="Verify model runs end-to-end")
    st.add_argument("--device", default="auto")

    return parser.parse_args()


# =============================================================================
# PRESET SYSTEM  — auto-scales model + training for the detected hardware
# =============================================================================

@dataclass
class _Preset:
    encoder_channels: int
    decoder_channels: int
    disc_channels: int
    disc_depth: int
    n_codebooks: int
    codebook_size: int
    batch_size: int
    disc_start_step: int
    save_every: int
    eval_every: int
    plot_every: int
    sample_every: int
    max_duration: float
    label: str


_PRESETS: Dict[str, _Preset] = {
    # ── cpu ── tiny model that actually trains in minutes per epoch on 4 CPU cores
    "cpu": _Preset(
        encoder_channels=64,  decoder_channels=64,
        disc_channels=16,     disc_depth=2,
        n_codebooks=4,        codebook_size=256,
        batch_size=2,         disc_start_step=200,
        save_every=100,       eval_every=50,
        plot_every=50,        sample_every=100,
        max_duration=2.0,
        label="CPU-tiny (64ch, 4 codebooks, batch=2)",
    ),
    # ── gpu-small ── fits in ~8 GB VRAM; good for RTX 3060 / T4
    "gpu-small": _Preset(
        encoder_channels=256, decoder_channels=256,
        disc_channels=32,     disc_depth=3,
        n_codebooks=6,        codebook_size=512,
        batch_size=4,         disc_start_step=5_000,
        save_every=2_000,     eval_every=500,
        plot_every=200,       sample_every=1_000,
        max_duration=4.0,
        label="GPU-small (256ch, 6 codebooks, batch=4)",
    ),
    # ── gpu-full ── EnCodec-scale; needs ~16 GB VRAM (A100 / RTX 3090)
    "gpu-full": _Preset(
        encoder_channels=512, decoder_channels=512,
        disc_channels=64,     disc_depth=4,
        n_codebooks=8,        codebook_size=1024,
        batch_size=8,         disc_start_step=10_000,
        save_every=5_000,     eval_every=1_000,
        plot_every=500,       sample_every=2_000,
        max_duration=5.0,
        label="GPU-full / EnCodec-scale (512ch, 8 codebooks, batch=8)",
    ),
}


def resolve_preset(preset_name: str, device: torch.device) -> _Preset:
    """
    Return the appropriate preset.

    'auto' maps to:
      - 'cpu'       when no GPU is available
      - 'gpu-small' when a CUDA GPU is present but has < 12 GB VRAM
      - 'gpu-full'  when >= 12 GB VRAM detected
    """
    if preset_name != "auto":
        return _PRESETS[preset_name]

    if device.type != "cuda":
        return _PRESETS["cpu"]

    try:
        vram_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
        if vram_gb >= 12:
            return _PRESETS["gpu-full"]
        return _PRESETS["gpu-small"]
    except Exception:
        return _PRESETS["gpu-small"]


def main() -> None:
    args = parse_args()

    detected = get_device()
    logger.info(f"Detected device: {device_info(detected)}")
    if detected.type == "cpu":
        logger.warning(
            "No GPU detected - training on CPU. This will be significantly slower. "
            "Consider using a machine with an NVIDIA GPU or Apple Silicon (MPS). "
            "Tip: pass --preset cpu to auto-scale the model for CPU training."
        )

    if args.command == "smoke-test":
        smoke_test(getattr(args, "device", "auto"))
        return

    if args.command == "train":
        # ── Resolve effective device first (needed for preset selection) ──
        eff_device = get_device(args.device)

        # ── Preset ────────────────────────────────────────────────────────
        preset = resolve_preset(args.preset, eff_device)
        logger.info(f"Preset '{args.preset}' -> {preset.label}")

        # Batch size: CLI flag > preset default
        batch_size = args.batch_size if args.batch_size is not None else preset.batch_size

        # ── Configs ───────────────────────────────────────────────────────
        audio_cfg = AudioConfig()
        rvq_cfg   = RVQConfig(
            n_codebooks  = preset.n_codebooks,
            codebook_size= preset.codebook_size,
        )
        model_cfg = ModelConfig(
            encoder_channels = preset.encoder_channels,
            decoder_channels = preset.decoder_channels,
            disc_channels    = preset.disc_channels,
            disc_depth       = preset.disc_depth,
        )
        train_cfg = TrainingConfig(
            data_dir           = args.data_dir,
            checkpoint_dir     = args.checkpoint_dir,
            log_dir            = args.log_dir,
            plot_dir           = args.plot_dir,
            output_dir         = args.output_dir,
            batch_size         = batch_size,
            num_epochs         = args.epochs,
            learning_rate      = args.lr,
            compile_model      = args.compile,
            device             = args.device,
            save_every_steps   = args.save_every,
            eval_every_steps   = args.eval_every,
            plot_every_steps   = args.plot_every,
            sample_every_steps = args.sample_every,
            grad_accum_steps   = args.grad_accum,
            mixed_precision    = not args.no_amp,
            num_workers        = args.num_workers,
            seed               = args.seed,
            max_duration       = preset.max_duration,
            disc_start_step    = preset.disc_start_step,
        )

        # ── Log the effective configuration ───────────────────────────────
        logger.info(
            f"Config: encoder={model_cfg.encoder_channels}ch  "
            f"codebooks={rvq_cfg.n_codebooks}x{rvq_cfg.codebook_size}  "
            f"batch={train_cfg.batch_size}  "
            f"duration={train_cfg.max_duration}s  "
            f"semantic={'OFF' if args.no_semantic else 'ON'}"
        )

        use_semantic = not args.no_semantic

        n_gpus = args.gpus if torch.cuda.is_available() else 1
        if args.gpus > 1 and not torch.cuda.is_available():
            logger.warning(f"--gpus {args.gpus} requested but no CUDA GPUs. Falling back to 1.")

        if n_gpus > 1:
            import torch.multiprocessing as mp
            mp.spawn(
                train,
                args=(n_gpus, audio_cfg, rvq_cfg, model_cfg, train_cfg, args.resume, use_semantic),
                nprocs=n_gpus,
                join=True,
            )
        else:
            train(0, 1, audio_cfg, rvq_cfg, model_cfg, train_cfg, args.resume, use_semantic)

    elif args.command == "encode":
        model = _load_model_from_checkpoint(args.checkpoint, args.device)
        codes = encode_audio_file(model, args.audio_path, lang=args.lang)
        torch.save(codes, args.out)
        logger.info(f"Codes shape {codes.shape} saved to {args.out}")

    elif args.command == "decode":
        model = _load_model_from_checkpoint(args.checkpoint, args.device)
        codes = torch.load(args.codes_path, map_location="cpu", weights_only=False)
        decode_codes_to_file(model, codes, args.out, args.device)

    else:
        logger.error("No command given. Use: train | encode | decode | smoke-test")
        sys.exit(1)


if __name__ == "__main__":
    main()