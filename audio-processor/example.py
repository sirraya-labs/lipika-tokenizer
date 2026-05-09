#!/usr/bin/env python3
"""Test the LATEST checkpoint with real audio - auto-finds newest checkpoint."""
import torch
import numpy as np
from pathlib import Path
from new import _load_model_from_checkpoint, encode_audio_file

# Find latest checkpoint
checkpoint_dir = Path('checkpoints')
ckpt_files = sorted(checkpoint_dir.glob('ckpt_step*.pt'))

if not ckpt_files:
    print("❌ No checkpoints found in checkpoints/")
    print("   Train first: python new.py")
    exit(1)

latest_ckpt = ckpt_files[-1]
print(f"📦 Loading latest checkpoint: {latest_ckpt.name}")
model = _load_model_from_checkpoint(str(latest_ckpt))

# Get codebook size from model
cb_size = model.rvq.codebooks[0].codebook_size
n_cb = model.rvq.n_codebooks

# Encode audio
print(f"🎵 Encoding speech_hindi.wav...")
codes = encode_audio_file(model, 'speech_hindi.wav', lang='hi')
codes_np = codes[0].cpu().numpy()
T = codes_np.shape[0]

# Results
print(f"\n{'='*60}")
print(f"  CODES FROM: {latest_ckpt.name}")
print(f"{'='*60}")
print(f"  Frames: {T}  |  Codebooks: {n_cb}  |  Codebook size: {cb_size}")
print(f"  Code range: [{codes_np.min()}, {codes_np.max()}]")
print(f"  Compression: 60x (189830 samples → {codes.numel()} tokens)")

# Codebook usage
print(f"\n  Codebook Usage:")
for cb in range(n_cb):
    flat = codes_np[:, cb]
    unique = len(np.unique(flat))
    usage = unique / cb_size * 100
    status = "✅" if usage > 40 else ("⚠️" if usage > 20 else "❌")
    print(f"    CB{cb}: {unique}/{cb_size} codes ({usage:.1f}%) {status}")

# First 20 frames
print(f"\n  First 20 frames:")
print(f"  {'Frame':>5} {'ms':>5} |", end="")
for cb in range(n_cb): print(f" CB{cb}", end="")
print(f"\n  {'-'*35}")
for t in range(min(20, T)):
    ms = t * 10
    c = codes_np[t]
    print(f"  {t:5d} {ms:5d} |", end="")
    for cb in range(n_cb): print(f" {int(c[cb]):3d}", end="")
    print()

# Top codes
print(f"\n  Top 3 codes per codebook:")
for cb in range(n_cb):
    flat = codes_np[:, cb].astype(int)
    counts = np.bincount(flat, minlength=cb_size)
    top3 = np.argsort(counts)[-3:][::-1]
    top_str = ", ".join([f"#{c}:{counts[c]}x ({counts[c]/T*100:.0f}%)" for c in top3 if counts[c] > 0])
    print(f"    CB{cb}: {top_str}")

# Save with checkpoint name
out_name = f"hindi_codes_{latest_ckpt.stem}.pt"
torch.save(codes, out_name)
print(f"\n  ✅ Saved to {out_name}")