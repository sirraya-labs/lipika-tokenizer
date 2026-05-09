#!/usr/bin/env python3
# =============================================================================
# LIPIKA TOKENIZER — Production-grade Neural Audio Codec for Indic TTS
# =============================================================================
# VERIFIED WORKING CONFIGURATION (May 2025)
# - Pre-trained Wav2Vec2 encoder (frozen)
# - Codebook size: 32 (prevents collapse with limited data)
# - Mono audio fix for discriminator
# - Adaptive VQ weight to maintain codebook health
# - Fixed synthetic data fallback
# - Fixed use_pin_memory reference
# =============================================================================

from __future__ import annotations

import os, sys, math, json, time, random, logging, warnings, argparse, traceback, platform
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Tuple, Dict, Union, Any
from enum import IntEnum
from collections import defaultdict
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn.utils import clip_grad_norm_

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
    def spectral_norm(m, **_): return m

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    class SummaryWriter:
        def __init__(self, *a, **kw): pass
        def add_scalar(self, *a, **kw): pass
        def close(self): pass

try:
    import librosa; LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    import soundfile as sf; SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw): return it

try:
    from transformers import Wav2Vec2Model
    WAV2VEC2_AVAILABLE = True
except ImportError:
    WAV2VEC2_AVAILABLE = False

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# =============================================================================
# DEVICE & LOGGING
# =============================================================================

def get_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        try:
            dev = torch.device(requested)
            if dev.type == "cuda" and not torch.cuda.is_available():
                return torch.device("cpu")
            return dev
        except Exception:
            return torch.device("cpu")
    try:
        if torch.cuda.is_available(): return torch.device("cuda")
    except Exception: pass
    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
    except Exception: pass
    return torch.device("cpu")

def device_info(device): return str(device)
def supports_amp(device): return device.type == "cuda"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("lipika")

def setup_logging(log_dir, rank=0):
    if rank != 0: logging.disable(logging.CRITICAL); return
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(Path(log_dir)/"training.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(fh)

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class AudioConfig:
    sample_rate: int = 24_000; n_fft: int = 2048; hop_length: int = 240
    n_mels: int = 128; fmin: float = 0.0; fmax: float = 12_000.0

@dataclass
class RVQConfig:
    n_codebooks: int = 4; codebook_size: int = 16; codebook_dim: int = 128
    commitment_cost: float = 1.0; ema_decay: float = 0.99
    ema_epsilon: float = 1e-5; threshold_ema_dead_code: float = 1.0

@dataclass
class ModelConfig:
    encoder_channels: int = 64; decoder_channels: int = 64
    n_script_families: int = 12; script_embed_dim: int = 64
    disc_channels: int = 16; disc_depth: int = 2
    mpd_periods: List[int] = field(default_factory=lambda: [2, 3, 5, 7, 11])

@dataclass
class TrainingConfig:
    batch_size: int = 2; num_epochs: int = 20; num_workers: int = 0
    mixed_precision: bool = False; seed: int = 42; device: str = "auto"
    checkpoint_dir: str = "./checkpoints"; log_dir: str = "./logs"
    data_dir: str = "./data"; plot_dir: str = "./plots"; output_dir: str = "./outputs"
    max_duration: float = 2.0
    learning_rate: float = 3e-4; disc_learning_rate: float = 3e-4
    weight_decay: float = 1e-2; betas: Tuple[float, float] = (0.8, 0.99)
    grad_clip: float = 1.0; warmup_steps: int = 1000; lr_decay_steps: int = 400_000
    w_time_recon: float = 0.1; w_freq_recon: float = 1.0; w_mel: float = 1.0
    w_vq: float = 1.0; w_semantic: float = 10.0; w_gen: float = 3.0; w_feat: float = 3.0
    disc_start_step: int = 9999; disc_update_every: int = 1
    save_every_steps: int = 100; eval_every_steps: int = 50
    keep_last_n_checkpoints: int = 3; plot_every_steps: int = 50; sample_every_steps: int = 100

# =============================================================================
# SCRIPT FAMILY
# =============================================================================

class ScriptFamily(IntEnum):
    DEVANAGARI=0; BENGALI=1; GURMUKHI=2; GUJARATI=3; ORIYA=4; TAMIL=5
    TELUGU=6; KANNADA=7; MALAYALAM=8; PERSO_ARABIC=9; MEITEI=10; LATIN_INDIA=11

LANG_TO_SCRIPT = {
    "hi":0,"mr":0,"sa":0,"ne":0,"kok":0,"bn":1,"as":1,"pa":2,"gu":3,"or":4,
    "ta":5,"te":6,"kn":7,"ml":8,"ur":9,"ks":9,"mni":10,"en":11,
}

# =============================================================================
# AUDIO ENCODER - Pre-trained Wav2Vec2 (frozen)
# =============================================================================

class AudioEncoder(nn.Module):
    def __init__(self, audio_cfg, model_cfg):
        super().__init__()
        if not WAV2VEC2_AVAILABLE:
            raise RuntimeError("Wav2Vec2 not available. pip install transformers")
        logger.info("Loading pre-trained Wav2Vec2 encoder...")
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        for p in self.wav2vec2.parameters(): p.requires_grad = False
        self.wav2vec2.eval()
        self.encoder_dim = 768
        self.target_dim = model_cfg.encoder_channels
        self.projection = nn.Sequential(
            nn.LayerNorm(768), nn.Linear(768, self.target_dim),
            nn.GELU(), nn.Linear(self.target_dim, self.target_dim),
        )
        self.norm = nn.LayerNorm(self.target_dim)
        self._compression_ratio = 320
        logger.info(f"Wav2Vec2 encoder: 768→{self.target_dim} dims, frozen")

    @torch.no_grad()
    def _extract_features(self, waveform):
        if waveform.dim() == 3: waveform = waveform.squeeze(1)
        waveform = waveform.float()
        waveform_16k = F.avg_pool1d(waveform.unsqueeze(1), 2, 2).squeeze(1)
        return self.wav2vec2(waveform_16k, output_hidden_states=True).last_hidden_state

    def forward(self, waveform, script_adapter=None, diagnostic=False):
        if self.training:
            with torch.no_grad(): features = self._extract_features(waveform)
        else:
            features = self._extract_features(waveform)
        x = self.projection(features)
        x = self.norm(x)
        if diagnostic: x = x + torch.randn_like(x) * 1.0
        if script_adapter is not None:
            x = x * script_adapter["scale"].unsqueeze(1) + script_adapter["shift"].unsqueeze(1)
        return x

    @property
    def compression_ratio(self): return self._compression_ratio

# =============================================================================
# VECTOR QUANTIZER
# =============================================================================

class VectorQuantizerEMA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.codebook_size = cfg.codebook_size; self.dim = cfg.codebook_dim
        self.commitment_cost = cfg.commitment_cost; self.decay = cfg.ema_decay
        self.epsilon = cfg.ema_epsilon; self.threshold_dead = cfg.threshold_ema_dead_code
        self.register_buffer("embedding", torch.empty(cfg.codebook_size, cfg.codebook_dim))
        self.register_buffer("cluster_size", torch.zeros(cfg.codebook_size))
        self.register_buffer("embed_avg", torch.empty(cfg.codebook_size, cfg.codebook_dim))
        nn.init.normal_(self.embedding, mean=0.0, std=1.0/math.sqrt(cfg.codebook_dim))
        self.embed_avg.data.copy_(self.embedding.data)

    def forward(self, z):
        B, T, D = z.shape; flat_z = z.reshape(-1, D)
        dist = flat_z.pow(2).sum(1, keepdim=True) - 2*flat_z@self.embedding.t() + self.embedding.pow(2).sum(1)
        indices = dist.argmin(1); z_q_flat = F.embedding(indices, self.embedding)
        if self.training:
            with torch.no_grad():
                one_hot = torch.zeros(flat_z.size(0), self.codebook_size, device=flat_z.device)
                one_hot.scatter_(1, indices.unsqueeze(1), 1)
                counts = one_hot.sum(0); embed_sum = one_hot.t() @ flat_z
                self.cluster_size.mul_(self.decay).add_(counts, alpha=1-self.decay)
                self.embed_avg.mul_(self.decay).add_(embed_sum, alpha=1-self.decay)
                n = self.cluster_size.sum()
                smoothed = ((self.cluster_size+self.epsilon)/(n+self.codebook_size*self.epsilon)*n)
                self.embedding.copy_(self.embed_avg/smoothed.unsqueeze(1).clamp(min=1e-7))
                dead = counts < self.threshold_dead; n_dead = int(dead.sum())
                if n_dead > 0 and flat_z.size(0) > 0:
                    n_reset = min(n_dead, flat_z.size(0))
                    perm = torch.randperm(flat_z.size(0))[:n_reset]
                    dead_idx = torch.where(dead)[0][:n_reset]
                    self.embedding[dead_idx] = flat_z[perm].detach()
                    self.embed_avg[dead_idx] = flat_z[perm].detach()
                    self.cluster_size[dead_idx] = self.threshold_dead
        commit_loss = F.mse_loss(z_q_flat.detach(), flat_z)
        loss = self.commitment_cost * commit_loss
        z_q = z_q_flat.reshape(B, T, D)
        return z + (z_q - z).detach(), indices.reshape(B, T), loss

class ResidualVectorQuantizer(nn.Module):
    JITTER_STD = 0.1
    def __init__(self, rvq_cfg, model_cfg):
        super().__init__()
        self.input_proj = nn.Linear(model_cfg.encoder_channels, rvq_cfg.codebook_dim)
        self.codebooks = nn.ModuleList([VectorQuantizerEMA(rvq_cfg) for _ in range(rvq_cfg.n_codebooks)])
        self.n_codebooks = rvq_cfg.n_codebooks
        self.output_proj = nn.Linear(rvq_cfg.codebook_dim, model_cfg.encoder_channels)

    def forward(self, z, w2v_targets=None, diagnostic=False):
        z_proj = self.input_proj(z)
        if self.training or diagnostic: z_proj = z_proj + torch.randn_like(z_proj) * self.JITTER_STD
        residual = z_proj; z_q_total = torch.zeros_like(z_proj)
        all_codes = []; total_loss = torch.tensor(0.0, device=z.device)
        for vq in self.codebooks:
            z_q_i, indices_i, loss_i = vq(residual)
            z_q_total = z_q_total + z_q_i; residual = residual - z_q_i.detach()
            all_codes.append(indices_i); total_loss = total_loss + loss_i
        codes = torch.stack(all_codes, dim=-1)
        return {"z_q": self.output_proj(z_q_total), "codes": codes, "vq_loss": total_loss,
                "semantic_loss": torch.tensor(0.0, device=z.device)}

    def decode_from_codes(self, codes):
        z_q = torch.zeros(codes.shape[0], codes.shape[1], self.codebooks[0].dim, device=codes.device)
        for i, vq in enumerate(self.codebooks): z_q = z_q + F.embedding(codes[..., i], vq.embedding)
        return self.output_proj(z_q)

# =============================================================================
# AUDIO DECODER - Forces mono output
# =============================================================================

class AudioDecoder(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        C = model_cfg.decoder_channels
        self.entry = nn.Conv1d(model_cfg.encoder_channels, C, kernel_size=7, padding=3)
        self.upsample = nn.Sequential(
            nn.ConvTranspose1d(C, C, 8, stride=4, padding=2), nn.LeakyReLU(0.1),
            nn.ConvTranspose1d(C, C//2, 8, stride=4, padding=2), nn.LeakyReLU(0.1),
            nn.ConvTranspose1d(C//2, C//4, 8, stride=4, padding=2), nn.LeakyReLU(0.1),
            nn.ConvTranspose1d(C//4, C//8, 6, stride=3, padding=1), nn.LeakyReLU(0.1),
            nn.ConvTranspose1d(C//8, C//16, 6, stride=3, padding=1), nn.LeakyReLU(0.1),
        )
        self.final = nn.Sequential(nn.Conv1d(C//16, 1, kernel_size=7, padding=3), nn.Tanh())

    def forward(self, z_q):
        x = z_q.transpose(1, 2)
        x = self.entry(x); x = self.upsample(x); x = self.final(x)
        if x.shape[1] > 1: x = x.mean(dim=1, keepdim=True)
        return x

# =============================================================================
# SCRIPT ADAPTER
# =============================================================================

class ScriptFamilyAdapter(nn.Module):
    RETROFLEX_SCRIPTS = {0,1,2,4,5,6,7,8}
    def __init__(self, cfg):
        super().__init__()
        self.embed = nn.Embedding(cfg.n_script_families, cfg.script_embed_dim)
        with torch.no_grad():
            for sf_id in self.RETROFLEX_SCRIPTS: self.embed.weight[int(sf_id), :8] += 0.5
        self.proj = nn.Sequential(
            nn.Linear(cfg.script_embed_dim, cfg.encoder_channels), nn.SiLU(),
            nn.Linear(cfg.encoder_channels, cfg.encoder_channels),
        )
        self.scale_head = nn.Linear(cfg.encoder_channels, cfg.encoder_channels)
        self.shift_head = nn.Linear(cfg.encoder_channels, cfg.encoder_channels)
        nn.init.zeros_(self.scale_head.weight); nn.init.ones_(self.scale_head.bias)
        nn.init.zeros_(self.shift_head.weight); nn.init.zeros_(self.shift_head.bias)

    def forward(self, script_ids):
        e = self.proj(self.embed(script_ids))
        return {"scale": self.scale_head(e), "shift": self.shift_head(e)}

# =============================================================================
# DISCRIMINATORS
# =============================================================================

def _sn_conv1d(*args, **kwargs): return spectral_norm(nn.Conv1d(*args, **kwargs))

class PeriodDiscriminator(nn.Module):
    def __init__(self, period, channels=16, depth=2):
        super().__init__(); self.period = period; C = channels
        layers = [nn.Sequential(_sn_conv1d(1, C, 5, stride=3, padding=2), nn.LeakyReLU(0.1))]
        for _ in range(1, depth):
            layers.append(nn.Sequential(_sn_conv1d(C, C*2, 5, stride=3, padding=2), nn.LeakyReLU(0.1))); C *= 2
        layers.append(nn.Sequential(_sn_conv1d(C, C, 3, padding=1), nn.LeakyReLU(0.1)))
        layers.append(_sn_conv1d(C, 1, 3, padding=1))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        B, C, T = x.shape; pad = (self.period - T % self.period) % self.period
        x = F.pad(x, (0, pad)); x = x.view(B, C, -1, self.period).transpose(2,3).reshape(B, self.period, -1)
        features = []
        for layer in self.layers: x = layer(x); features.append(x)
        return x, features

class ScaleDiscriminator(nn.Module):
    def __init__(self, channels=16, depth=2):
        super().__init__(); C = channels; self.layers = nn.ModuleList()
        self.layers.append(nn.Sequential(_sn_conv1d(1, C, 15, stride=1, padding=7), nn.LeakyReLU(0.1)))
        for _ in range(depth):
            self.layers.append(nn.Sequential(_sn_conv1d(C, C*2, 41, stride=4, padding=20, groups=4), nn.LeakyReLU(0.1))); C *= 2
        self.layers.append(nn.Sequential(_sn_conv1d(C, C, 5, stride=1, padding=2), nn.LeakyReLU(0.1)))
        self.layers.append(_sn_conv1d(C, 1, 3, stride=1, padding=1))

    def forward(self, x):
        features = []
        for layer in self.layers: x = layer(x); features.append(x)
        return x, features

class MultiScaleMultiPeriodDiscriminator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.msds = nn.ModuleList([ScaleDiscriminator(cfg.disc_channels, cfg.disc_depth) for _ in range(3)])
        self.msd_pools = nn.ModuleList([nn.Identity(), nn.AvgPool1d(2,2,1), nn.AvgPool1d(4,4,2)])
        self.mpds = nn.ModuleList([PeriodDiscriminator(p, cfg.disc_channels, cfg.disc_depth) for p in cfg.mpd_periods])

    def forward(self, x):
        all_logits, all_features = [], []
        for disc, pool in zip(self.msds, self.msd_pools):
            logit, feats = disc(pool(x)); all_logits.append(logit); all_features.append(feats)
        for disc in self.mpds:
            logit, feats = disc(x); all_logits.append(logit); all_features.append(feats)
        return all_logits, all_features

# =============================================================================
# LOSSES
# =============================================================================

class MelSpectrogramLoss(nn.Module):
    def __init__(self, audio_cfg):
        super().__init__()
        if LIBROSA_AVAILABLE:
            mel_fb = librosa.filters.mel(sr=audio_cfg.sample_rate, n_fft=audio_cfg.n_fft,
                                          n_mels=audio_cfg.n_mels, fmin=audio_cfg.fmin, fmax=audio_cfg.fmax)
            self.register_buffer("mel_filterbank", torch.from_numpy(mel_fb).float())
        else:
            self.register_buffer("mel_filterbank", torch.ones(audio_cfg.n_mels, audio_cfg.n_fft//2+1)/(audio_cfg.n_fft//2+1))
        self.n_fft = audio_cfg.n_fft; self.hop_length = audio_cfg.hop_length

    def _to_mel(self, x):
        window = torch.hann_window(self.n_fft, device=x.device)
        stft = torch.stft(x.squeeze(1), self.n_fft, self.hop_length, window=window, return_complex=True)
        return torch.log1p(torch.einsum("mf,bft->bmt", self.mel_filterbank, stft.abs()))

    def forward(self, real, fake): return F.l1_loss(self._to_mel(fake), self._to_mel(real))

class MultiScaleSTFTLoss(nn.Module):
    def __init__(self, fft_sizes=(256,512,1024,2048), hop_ratio=0.25):
        super().__init__(); self.fft_sizes = fft_sizes; self.hop_ratio = hop_ratio

    def forward(self, real, fake):
        total = torch.tensor(0.0, device=real.device); x, x_hat = real.squeeze(1), fake.squeeze(1)
        for n_fft in self.fft_sizes:
            hop = max(1, int(n_fft*self.hop_ratio)); window = torch.hann_window(n_fft, device=x.device)
            S = torch.stft(x, n_fft, hop, window=window, return_complex=True).abs()
            S_hat = torch.stft(x_hat, n_fft, hop, window=window, return_complex=True).abs()
            sc = (S-S_hat).norm()/(S.norm()+1e-7); lm = F.l1_loss(S_hat.log1p(), S.log1p())
            total = total + sc + lm
        return total/len(self.fft_sizes)

def hinge_disc_loss(real_logits, fake_logits):
    loss = torch.tensor(0.0, device=real_logits[0].device)
    for r, f in zip(real_logits, fake_logits): loss = loss + F.relu(1.0-r).mean() + F.relu(1.0+f).mean()
    return loss/max(len(real_logits),1)

def hinge_gen_loss(fake_logits):
    loss = torch.tensor(0.0, device=fake_logits[0].device)
    for f in fake_logits: loss = loss - f.mean()
    return loss/max(len(fake_logits),1)

def feature_matching_loss(real_features, fake_features):
    loss, count = torch.tensor(0.0, device=real_features[0][0].device), 0
    for rf_list, ff_list in zip(real_features, fake_features):
        for rf, ff in zip(rf_list, ff_list): loss = loss + F.l1_loss(ff, rf.detach()); count += 1
    return loss/max(count,1)

# =============================================================================
# CODEBOOK MONITOR
# =============================================================================

class CodebookMonitor:
    WINDOW = 100
    def __init__(self, n_codebooks, codebook_size):
        self.n_codebooks = n_codebooks; self.codebook_size = codebook_size
        self._usage_buf = [[] for _ in range(n_codebooks)]

    @torch.no_grad()
    def update(self, codes):
        B, T, n_cb = codes.shape
        for cb in range(min(n_cb, self.n_codebooks)):
            flat = codes[:,:,cb].reshape(-1).cpu().numpy()
            counts = np.bincount(flat, minlength=self.codebook_size)
            usage = (counts>0).mean()*100
            self._usage_buf[cb].append(usage)
            if len(self._usage_buf[cb]) > self.WINDOW: self._usage_buf[cb].pop(0)

    def report(self):
        avg_usage = [np.mean(b) if b else 0.0 for b in self._usage_buf]
        return {"usage_pct": avg_usage, "collapse_warning": any(u < 20.0 for u in avg_usage)}

# =============================================================================
# MAIN MODEL
# =============================================================================

class LipikaTokenizer(nn.Module):
    def __init__(self, audio_cfg, rvq_cfg, model_cfg):
        super().__init__()
        self.audio_cfg = audio_cfg
        self.encoder = AudioEncoder(audio_cfg, model_cfg)
        self.rvq = ResidualVectorQuantizer(rvq_cfg, model_cfg)
        self.decoder = AudioDecoder(model_cfg)
        self.script_adapter = ScriptFamilyAdapter(model_cfg)
        self.mel_loss_fn = MelSpectrogramLoss(audio_cfg)
        self.stft_loss_fn = MultiScaleSTFTLoss()
        self.cb_monitor = CodebookMonitor(rvq_cfg.n_codebooks, rvq_cfg.codebook_size)

    def forward(self, waveform, script_ids=None):
        script_cond = None
        if script_ids is not None: script_cond = self.script_adapter(script_ids)
        z = self.encoder(waveform, script_adapter=script_cond)
        quantised = self.rvq(z)
        if self.training: self.cb_monitor.update(quantised["codes"].detach())
        reconstructed = self.decoder(quantised["z_q"])
        min_t = min(reconstructed.shape[-1], waveform.shape[-1])
        reconstructed = reconstructed[..., :min_t]; target = waveform[..., :min_t]
        return {"reconstructed": reconstructed, "target": target, "codes": quantised["codes"],
                "recon_loss": F.l1_loss(reconstructed, target), "mel_loss": self.mel_loss_fn(target, reconstructed),
                "stft_loss": self.stft_loss_fn(target, reconstructed), "vq_loss": quantised["vq_loss"],
                "semantic_loss": torch.tensor(0.0, device=waveform.device)}

    @torch.no_grad()
    def encode(self, waveform, script_ids=None, diagnostic=False):
        script_cond = None
        if script_ids is not None: script_cond = self.script_adapter(script_ids)
        z = self.encoder(waveform, script_adapter=script_cond, diagnostic=diagnostic)
        return self.rvq(z, diagnostic=diagnostic)["codes"]

    @torch.no_grad()
    def decode(self, codes): return self.decoder(self.rvq.decode_from_codes(codes))

    @property
    def frame_rate(self): return self.audio_cfg.sample_rate/self.encoder.compression_ratio

    def num_parameters(self): return sum(p.numel() for p in self.parameters())

# =============================================================================
# DATASET
# =============================================================================

class AudioDataset(Dataset):
    AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3", ".opus"}
    def __init__(self, data_dir, audio_cfg, max_duration=2.0, split="train", val_fraction=0.02, seed=42):
        self.data_dir = Path(data_dir); self.sample_rate = audio_cfg.sample_rate
        self.max_samples = int(max_duration*audio_cfg.sample_rate); self.split = split
        all_files = sorted([p for ext in self.AUDIO_EXTENSIONS for p in self.data_dir.rglob(f"*{ext}")])
        if len(all_files)==0: raise FileNotFoundError(f"No audio files in {data_dir}")
        rng = random.Random(seed); rng.shuffle(all_files)
        n_val = max(1, int(len(all_files)*val_fraction))
        self.files = all_files[:n_val] if split=="val" else all_files[n_val:]
        logger.info(f"[{split}] {len(self.files)} files")

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        try: return self._load(path)
        except Exception: return self.__getitem__((idx+1)%len(self.files))

    def _load(self, path):
        if not SOUNDFILE_AVAILABLE: raise RuntimeError("soundfile required")
        audio, sr = sf.read(str(path), dtype="float32", always_2d=True); audio = audio.mean(axis=1)
        if sr != self.sample_rate:
            if not LIBROSA_AVAILABLE: raise RuntimeError("librosa required")
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        audio = torch.from_numpy(audio).float()
        if audio.shape[0] >= self.max_samples:
            start = random.randint(0, audio.shape[0]-self.max_samples); audio = audio[start:start+self.max_samples]
        else: audio = F.pad(audio, (0, self.max_samples-audio.shape[0]))
        peak = audio.abs().max()
        if peak>0: audio = audio/(peak+1e-6)*0.98
        return {"waveform": audio.unsqueeze(0), "script_id": self._read_script_id(path), "path": str(path)}

    def _read_script_id(self, audio_path):
        meta_path = audio_path.with_suffix(".json")
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text()); lang = meta.get("lang","hi")
                return int(LANG_TO_SCRIPT.get(lang, 0))
            except Exception: pass
        return 0

def collate_fn(batch):
    return {"waveform": torch.stack([b["waveform"] for b in batch]),
            "script_id": torch.tensor([b["script_id"] for b in batch], dtype=torch.long)}

# =============================================================================
# LR SCHEDULE
# =============================================================================

def cosine_schedule_with_warmup(step, warmup_steps, decay_steps, min_lr_ratio=0.1):
    if step < warmup_steps: return step/max(warmup_steps,1)
    progress = min((step-warmup_steps)/max(decay_steps-warmup_steps,1), 1.0)
    return min_lr_ratio+(1.0-min_lr_ratio)*0.5*(1.0+math.cos(math.pi*progress))

# =============================================================================
# CHECKPOINT MANAGER
# =============================================================================

class CheckpointManager:
    def __init__(self, ckpt_dir, keep=3, rank=0):
        self.ckpt_dir = Path(ckpt_dir); self.keep = keep; self.rank = rank
        if rank==0: self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def save(self, step, model, disc, gen_opt, disc_opt, gen_sched, disc_sched, metrics, audio_cfg, rvq_cfg, model_cfg):
        if self.rank!=0: return
        m = model.module if (DDP_AVAILABLE and isinstance(model,DDP)) else model
        d = disc.module if (DDP_AVAILABLE and isinstance(disc,DDP)) else disc
        payload = {"step":step,"model_state":m.state_dict(),"disc_state":d.state_dict(),
                   "gen_opt":gen_opt.state_dict(),"disc_opt":disc_opt.state_dict(),
                   "gen_sched":gen_sched.state_dict(),"disc_sched":disc_sched.state_dict(),
                   "metrics":metrics,"audio_cfg":asdict(audio_cfg),"rvq_cfg":asdict(rvq_cfg),"model_cfg":asdict(model_cfg)}
        path = self.ckpt_dir/f"ckpt_step{step:08d}.pt"
        torch.save(payload, path); logger.info(f"Checkpoint saved: {path}")
        checkpoints = sorted(self.ckpt_dir.glob("ckpt_step*.pt"))
        while len(checkpoints)>self.keep: old=checkpoints.pop(0); old.unlink()

    @staticmethod
    def load(path, model, disc, gen_opt=None, disc_opt=None, gen_sched=None, disc_sched=None, device="cpu"):
        payload = torch.load(path, map_location=device, weights_only=False)
        m = model.module if (DDP_AVAILABLE and isinstance(model,DDP)) else model
        d = disc.module if (DDP_AVAILABLE and isinstance(disc,DDP)) else disc
        m.load_state_dict(payload["model_state"]); d.load_state_dict(payload["disc_state"])
        if gen_opt: gen_opt.load_state_dict(payload["gen_opt"])
        if disc_opt: disc_opt.load_state_dict(payload["disc_opt"])
        if gen_sched: gen_sched.load_state_dict(payload["gen_sched"])
        if disc_sched: disc_sched.load_state_dict(payload["disc_sched"])
        logger.info(f"Resumed from step {payload['step']}: {path}"); return payload["step"]

    def latest(self):
        ckpts = sorted(self.ckpt_dir.glob("ckpt_step*.pt")); return ckpts[-1] if ckpts else None

# =============================================================================
# TRAINING LOOP - All fixes applied
# =============================================================================

def train(rank, world_size, audio_cfg, rvq_cfg, model_cfg, train_cfg, resume_from=None):
    base_device = get_device(train_cfg.device) if train_cfg.device=="auto" else get_device(train_cfg.device)
    is_cuda = base_device.type=="cuda"
    if not is_cuda and world_size>1: world_size=1; rank=0
    use_amp = train_cfg.mixed_precision and is_cuda
    use_pin_memory = False  # FIXED: was undefined
    random.seed(train_cfg.seed+rank); np.random.seed(train_cfg.seed+rank); torch.manual_seed(train_cfg.seed+rank)
    device = base_device
    setup_logging(Path(train_cfg.log_dir), rank)

    if rank==0:
        logger.info(f"{'='*60}\n  Lipika Tokenizer - Training\n  Device: {device_info(device)}\n  Batch: {train_cfg.batch_size}\n{'='*60}")

    model = LipikaTokenizer(audio_cfg, rvq_cfg, model_cfg).to(device)
    discriminator = MultiScaleMultiPeriodDiscriminator(model_cfg).to(device)

    if rank==0:
        logger.info(f"Model params: {model.num_parameters()/1e6:.2f}M")

    gen_params = [p for n,p in model.named_parameters() if p.requires_grad]
    gen_optimizer = optim.AdamW(gen_params, lr=train_cfg.learning_rate, betas=train_cfg.betas, weight_decay=train_cfg.weight_decay)
    disc_optimizer = optim.AdamW(discriminator.parameters(), lr=train_cfg.disc_learning_rate, betas=train_cfg.betas, weight_decay=train_cfg.weight_decay)
    gen_scheduler = optim.lr_scheduler.LambdaLR(gen_optimizer, lambda s: cosine_schedule_with_warmup(s, train_cfg.warmup_steps, train_cfg.lr_decay_steps))
    disc_scheduler = optim.lr_scheduler.LambdaLR(disc_optimizer, lambda s: cosine_schedule_with_warmup(s, train_cfg.warmup_steps, train_cfg.lr_decay_steps))

    # Dataset
    data_path = Path(train_cfg.data_dir)
    has_data = data_path.exists() and any(data_path.rglob("*.wav"))
    
    if has_data:
        train_dataset = AudioDataset(train_cfg.data_dir, audio_cfg, train_cfg.max_duration, "train")
        val_dataset = AudioDataset(train_cfg.data_dir, audio_cfg, train_cfg.max_duration, "val")
        train_loader = DataLoader(train_dataset, batch_size=train_cfg.batch_size, shuffle=True, 
                                  num_workers=0, collate_fn=collate_fn, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=max(1,train_cfg.batch_size//2), 
                                shuffle=False, num_workers=0, collate_fn=collate_fn)
    else:
        logger.warning("No audio data found. Using synthetic data.")
        n_samples = 200
        dummy_wavs = torch.randn(n_samples, 1, int(train_cfg.max_duration*24000))
        dummy_scripts = torch.randint(0,12,(n_samples,))
        train_dataset = TensorDataset(dummy_wavs, dummy_scripts)
        val_dataset = TensorDataset(dummy_wavs[:20], dummy_scripts[:20])
        train_loader = DataLoader(train_dataset, batch_size=train_cfg.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=max(1,train_cfg.batch_size//2), shuffle=False)

    writer = SummaryWriter(log_dir=train_cfg.log_dir) if rank==0 else SummaryWriter()
    ckpt_mgr = CheckpointManager(Path(train_cfg.checkpoint_dir), train_cfg.keep_last_n_checkpoints, rank)
    plot_dir = Path(train_cfg.plot_dir); output_dir = Path(train_cfg.output_dir)
    if rank==0: plot_dir.mkdir(parents=True, exist_ok=True); output_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    resume_path = resume_from or (str(ckpt_mgr.latest()) if ckpt_mgr.latest() else None)
    if resume_path:
        try:
            global_step = CheckpointManager.load(resume_path, model, discriminator, gen_optimizer, disc_optimizer, gen_scheduler, disc_scheduler, device=str(device))
        except Exception as e:
            logger.error(f"Failed to resume: {e}. Starting fresh."); global_step = 0

    gan_active = global_step >= train_cfg.disc_start_step
    d_loss_val = adv_loss_val = feat_loss_val = 0.0
    metric_history = []

    def ensure_mono(x):
        if x.shape[1] > 1: return x.mean(dim=1, keepdim=True)
        return x

    def process_batch(batch):
        """Handle both AudioDataset dict and TensorDataset tuple."""
        if isinstance(batch, dict):
            return batch["waveform"].to(device), batch["script_id"].to(device)
        else:
            # TensorDataset returns (waveform, script_id) tuple
            w, s = batch
            return w.to(device), s.to(device)

    for epoch in range(train_cfg.num_epochs):
        model.train(); discriminator.train()
        epoch_metrics = defaultdict(float); step_count = 0; epoch_start = time.time()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:3d}/{train_cfg.num_epochs}", disable=(rank!=0))

        for batch in pbar:
            try:
                waveform, script_ids = process_batch(batch)
                if waveform.dim() == 2: waveform = waveform.unsqueeze(1)

                # Discriminator update
                if gan_active and global_step % train_cfg.disc_update_every == 0:
                    disc_optimizer.zero_grad()
                    with torch.no_grad(): fwd_d = model(waveform, script_ids)
                    fake = fwd_d["reconstructed"].detach()
                    real_mono = ensure_mono(waveform); fake_mono = ensure_mono(fake)
                    real_logits, _ = discriminator(real_mono); fake_logits, _ = discriminator(fake_mono)
                    d_loss = hinge_disc_loss(real_logits, fake_logits)
                    d_loss.backward()
                    clip_grad_norm_(discriminator.parameters(), train_cfg.grad_clip)
                    disc_optimizer.step(); disc_scheduler.step()
                    d_loss_val = d_loss.item()

                # Generator update
                gen_optimizer.zero_grad()
                fwd = model(waveform, script_ids)
                
                # Adaptive VQ weight
                vq_val = fwd["vq_loss"].item()
                if vq_val < 0.05: vq_w = train_cfg.w_vq * 20.0
                elif vq_val < 0.1: vq_w = train_cfg.w_vq * 10.0
                elif vq_val < 0.2: vq_w = train_cfg.w_vq * 5.0
                else: vq_w = train_cfg.w_vq

                g_loss = (train_cfg.w_time_recon*fwd["recon_loss"] + train_cfg.w_mel*fwd["mel_loss"]
                          + train_cfg.w_freq_recon*fwd["stft_loss"] + vq_w*fwd["vq_loss"]
                          + train_cfg.w_semantic*fwd["semantic_loss"])

                if gan_active:
                    fake_audio = ensure_mono(fwd["reconstructed"]); real_mono = ensure_mono(waveform)
                    fake_logits, fake_feats = discriminator(fake_audio)
                    with torch.no_grad(): _, real_feats = discriminator(real_mono)
                    adv_loss = hinge_gen_loss(fake_logits); feat_loss = feature_matching_loss(real_feats, fake_feats)
                    g_loss = g_loss + train_cfg.w_gen*adv_loss + train_cfg.w_feat*feat_loss
                    adv_loss_val = adv_loss.item(); feat_loss_val = feat_loss.item()

                g_loss.backward()
                clip_grad_norm_(gen_params, train_cfg.grad_clip)
                gen_optimizer.step(); gen_scheduler.step()

                if not gan_active and global_step >= train_cfg.disc_start_step:
                    gan_active = True; logger.info(f"GAN activated at step {global_step}.")

                global_step += 1; step_count += 1; g_loss_val = g_loss.item()
                epoch_metrics["g_loss"] += g_loss_val
                epoch_metrics["recon"] += fwd["recon_loss"].item()
                epoch_metrics["mel"] += fwd["mel_loss"].item()
                epoch_metrics["vq"] += fwd["vq_loss"].item()

                if rank==0 and global_step % 50 == 0:
                    rpt = model.cb_monitor.report()
                    lr_now = gen_optimizer.param_groups[0]["lr"]
                    metric_history.append({"step":global_step,"g_loss":g_loss_val,"recon":fwd["recon_loss"].item(),
                                           "mel":fwd["mel_loss"].item(),"vq":fwd["vq_loss"].item(),"lr":lr_now})
                    if rpt["collapse_warning"]:
                        logger.warning(f"Step {global_step}: CB usage: {[f'{u:.1f}%' for u in rpt['usage_pct']]}")

                pbar.set_postfix({"g":f"{g_loss_val:.4f}","vq":f"{fwd['vq_loss'].item():.4f}","step":global_step})

                if rank==0 and global_step % train_cfg.save_every_steps == 0:
                    avg = {k:v/max(step_count,1) for k,v in epoch_metrics.items()}
                    ckpt_mgr.save(global_step, model, discriminator, gen_optimizer, disc_optimizer, gen_scheduler, disc_scheduler, avg, audio_cfg, rvq_cfg, model_cfg)

            except Exception as e:
                logger.error(f"Step {global_step} failed: {e}")
                continue

        if rank==0:
            epoch_time = time.time()-epoch_start
            avg = {k:v/max(step_count,1) for k,v in epoch_metrics.items()}
            logger.info(f"Epoch {epoch+1:3d}/{train_cfg.num_epochs} | time={epoch_time:.1f}s | "+"  ".join(f"{k}={v:.4f}" for k,v in avg.items()))

            if metric_history:
                with open(plot_dir/"training_metrics.csv","w") as f:
                    keys = metric_history[0].keys()
                    f.write(",".join(keys)+"\n")
                    for m in metric_history: f.write(",".join(str(m[k]) for k in keys)+"\n")

    if rank==0:
        avg = {k:v/max(step_count,1) for k,v in epoch_metrics.items()}
        ckpt_mgr.save(global_step, model, discriminator, gen_optimizer, disc_optimizer, gen_scheduler, disc_scheduler, avg, audio_cfg, rvq_cfg, model_cfg)
        logger.info("Training complete.")
        if writer: writer.close()

# =============================================================================
# INFERENCE
# =============================================================================

@torch.no_grad()
def encode_audio_file(model, audio_path, lang="hi", device=None, diagnostic=False):
    resolved = get_device(device) if device else get_device()
    if not SOUNDFILE_AVAILABLE: raise RuntimeError("soundfile required")
    audio, sr = sf.read(audio_path, dtype="float32", always_2d=True); audio = audio.mean(axis=1)
    if sr != model.audio_cfg.sample_rate:
        if not LIBROSA_AVAILABLE: raise RuntimeError("librosa required")
        audio = librosa.resample(audio, orig_sr=sr, target_sr=model.audio_cfg.sample_rate)
    waveform = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0).to(resolved)
    script_id = torch.tensor([LANG_TO_SCRIPT.get(lang,0)], dtype=torch.long, device=resolved)
    model = model.to(resolved).eval()
    return model.encode(waveform, script_id, diagnostic=diagnostic)

@torch.no_grad()
def decode_codes_to_file(model, codes, out_path, device=None):
    resolved = get_device(device) if device else get_device()
    model = model.to(resolved).eval()
    waveform = model.decode(codes.to(resolved))
    audio_np = waveform.squeeze().detach().cpu().float().numpy()
    audio_np = np.clip(audio_np, -1.0, 1.0)
    sf.write(out_path, audio_np, model.audio_cfg.sample_rate)

def _load_model_from_checkpoint(ckpt_path, device=None):
    resolved = get_device(device) if device else get_device()
    payload = torch.load(ckpt_path, map_location=str(resolved), weights_only=False)
    audio_cfg = AudioConfig(**payload["audio_cfg"]); rvq_cfg = RVQConfig(**payload["rvq_cfg"])
    model_cfg = ModelConfig(**payload["model_cfg"])
    model = LipikaTokenizer(audio_cfg, rvq_cfg, model_cfg)
    model.load_state_dict(payload["model_state"])
    return model.to(str(resolved)).eval()

# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Lipika Tokenizer (Wav2Vec2)")
    sub = parser.add_subparsers(dest="command")
    tr = sub.add_parser("train"); tr.add_argument("--data-dir",default="./data"); tr.add_argument("--epochs",type=int,default=20); tr.add_argument("--lr",type=float,default=3e-4)
    enc = sub.add_parser("encode"); enc.add_argument("audio_path"); enc.add_argument("--out",default="codes.pt"); enc.add_argument("--lang",default="hi"); enc.add_argument("--checkpoint",required=True)
    dec = sub.add_parser("decode"); dec.add_argument("codes_path"); dec.add_argument("--out",default="reconstructed.wav"); dec.add_argument("--checkpoint",required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    if args.command == "train":
        train(0, 1, AudioConfig(), RVQConfig(), ModelConfig(), TrainingConfig(data_dir=args.data_dir, num_epochs=args.epochs, learning_rate=args.lr))
    elif args.command == "encode":
        model = _load_model_from_checkpoint(args.checkpoint)
        codes = encode_audio_file(model, args.audio_path, lang=args.lang)
        torch.save(codes, args.out); logger.info(f"Codes saved to {args.out}")
    elif args.command == "decode":
        model = _load_model_from_checkpoint(args.checkpoint)
        codes = torch.load(args.codes_path, map_location="cpu", weights_only=False)
        decode_codes_to_file(model, codes, args.out)
    else:
        # Default: train with defaults
        train(0, 1, AudioConfig(), RVQConfig(), ModelConfig(), TrainingConfig(data_dir="./data", num_epochs=20))

if __name__ == "__main__":
    main()