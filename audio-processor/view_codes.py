#!/usr/bin/env python3
"""
============================================================================
LIPIKA CODE VIEWER - See Your Audio as Discrete Tokens
============================================================================

This script encodes audio files and shows you exactly what the codes look like,
with beautiful formatting and analysis.

Usage:
    python view_codes.py --checkpoint checkpoints/ckpt_step00000201.pt --audio my_audio.wav
    python view_codes.py --checkpoint checkpoints/ckpt_step00000201.pt --audio-dir ./data
============================================================================
"""

import os
import sys
import json
import torch
import numpy as np
import soundfile as sf
import librosa
import argparse
from pathlib import Path
from collections import Counter

# Import Lipika
from lipika import (
    LipikaTokenizer,
    AudioConfig,
    RVQConfig,
    ModelConfig,
    get_device,
    device_info,
    LANG_TO_SCRIPT,
    ScriptFamily,
    _load_model_from_checkpoint,
    _PRESETS
)

# Terminal colors for pretty output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
    
    # Heatmap colors for codebooks
    HEATMAP = [
        '\033[48;5;196m',  # Red (very frequent)
        '\033[48;5;202m',
        '\033[48;5;208m',
        '\033[48;5;214m',
        '\033[48;5;220m',
        '\033[48;5;226m',  # Yellow
        '\033[48;5;190m',
        '\033[48;5;154m',
        '\033[48;5;118m',
        '\033[48;5;82m',   # Green
        '\033[48;5;46m',   # Bright green (very rare)
    ]


def display_codes_grid(codes, codebook_names=None, max_frames=50):
    """
    Display codes as a colored grid.
    
    Args:
        codes: (1, T, n_codebooks) tensor
        codebook_names: list of names for each codebook
        max_frames: maximum frames to display
    """
    codes_np = codes[0].cpu().numpy()  # (T, n_codebooks)
    T, n_cb = codes_np.shape
    codebook_size = int(codes_np.max()) + 1
    
    if codebook_names is None:
        codebook_names = [f"CB{i}" for i in range(n_cb)]
    
    # Truncate if too long
    if T > max_frames:
        codes_np = codes_np[:max_frames]
        print(f"{Colors.YELLOW}Showing first {max_frames} of {T} frames{Colors.END}")
    
    print(f"\n{Colors.BOLD}{'='*120}{Colors.END}")
    print(f"{Colors.BOLD}  AUDIO TOKEN MAP - {T} frames × {n_cb} codebooks{Colors.END}")
    print(f"{Colors.BOLD}{'='*120}{Colors.END}")
    
    # Header row: time indices
    print(f"\n{Colors.CYAN}Time →{Colors.END}", end="")
    for t in range(min(T, max_frames)):
        if t % 10 == 0:
            print(f"{Colors.CYAN}{t:4d}{Colors.END}", end="")
        elif t % 5 == 0:
            print(f"{Colors.CYAN}  · {Colors.END}", end="")
        else:
            print("    ", end="")
    print()
    
    # Time ruler
    time_ms = [f"{t*10}" for t in range(0, min(T*10, max_frames*10), 50)]  # ms
    print(f"{Colors.CYAN}ms →  {Colors.END}", end="")
    for t in range(min(T, max_frames)):
        if t % 5 == 0:
            ms = t * 10
            print(f"{Colors.CYAN}{ms:3d} {Colors.END}", end="")
        else:
            print("     ", end="")
    print()
    
    # Separator
    print("    " + "─" * (min(T, max_frames) * 4 + 4))
    
    # Each codebook row
    for cb in range(n_cb):
        print(f"{Colors.BOLD}{codebook_names[cb]:>3} {Colors.END}", end="│")
        
        for t in range(min(T, max_frames)):
            code_val = int(codes_np[t, cb])
            heat_idx = min(code_val * 10 // codebook_size, 10)
            color = Colors.HEATMAP[heat_idx]
            print(f"{color}{code_val:3d}{Colors.END}", end=" ")
        
        print()
    
    print("    " + "─" * (min(T, max_frames) * 4 + 4))
    print(f"\n{Colors.CYAN}Legend: Red=low code ID, Green=high code ID{Colors.END}")


def display_codebook_stats(codes, codebook_names=None):
    """Display per-codebook statistics."""
    codes_np = codes[0].cpu().numpy()
    T, n_cb = codes_np.shape
    codebook_size = int(codes_np.max()) + 1
    
    if codebook_names is None:
        codebook_names = [f"CB{i}" for i in range(n_cb)]
    
    print(f"\n{Colors.BOLD}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}  CODEBOOK STATISTICS{Colors.END}")
    print(f"{Colors.BOLD}{'='*80}{Colors.END}")
    
    stats = []
    for cb in range(n_cb):
        codes_cb = codes_np[:, cb]
        counter = Counter(codes_cb)
        unique = len(counter)
        usage_pct = unique / codebook_size * 100
        
        # Entropy and perplexity
        counts = np.bincount(codes_cb, minlength=codebook_size)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log2(probs))
        max_entropy = np.log2(codebook_size)
        perplexity = 2 ** entropy
        
        stats.append({
            'cb': cb,
            'unique': unique,
            'usage_pct': usage_pct,
            'entropy': entropy,
            'max_entropy': max_entropy,
            'perplexity': perplexity,
            'top_codes': counter.most_common(5),
            'unused': codebook_size - unique
        })
    
    # Display table
    print(f"\n{'CB':>4} {'Unique':>8} {'Usage %':>8} {'Entropy':>10} {'Perplexity':>12} {'Unused':>8} {'Status'}")
    print("-" * 70)
    
    for s in stats:
        status = "✅" if s['usage_pct'] > 50 else ("⚠️" if s['usage_pct'] > 20 else "❌")
        print(f"  {s['cb']:1d}  {s['unique']:8d} {s['usage_pct']:7.1f}% {s['entropy']:9.2f} {s['perplexity']:11.1f} {s['unused']:8d}  {status}")
    
    print("-" * 70)
    
    # Top codes per codebook
    print(f"\n{Colors.BOLD}Top 5 most-used codes per codebook:{Colors.END}")
    for s in stats:
        top_str = ", ".join([f"#{c[0]:3d}({c[1]:4d}×)" for c in s['top_codes']])
        print(f"  {codebook_names[s['cb']]}: {top_str}")
    
    return stats


def display_code_sequence(codes, start_frame=0, n_frames=30):
    """Display codes as a sequence with time alignment."""
    codes_np = codes[0].cpu().numpy()
    T, n_cb = codes_np.shape
    
    end_frame = min(start_frame + n_frames, T)
    
    print(f"\n{Colors.BOLD}{'='*100}{Colors.END}")
    print(f"{Colors.BOLD}  CODE SEQUENCE - Frames {start_frame} to {end_frame-1}{Colors.END}")
    print(f"{Colors.BOLD}{'='*100}{Colors.END}")
    
    print(f"\n{'Frame':>6} {'ms':>5} |", end="")
    for cb in range(n_cb):
        print(f" CB{cb}", end="")
    print(" | Combined Token")
    print("-" * (20 + n_cb * 5 + 18))
    
    for t in range(start_frame, end_frame):
        codes_t = codes_np[t]
        ms = t * 10  # 10ms per frame at 100Hz
        
        # Combined token (for language modeling)
        combined = 0
        for cb in range(n_cb):
            combined = combined * int(codes_np.max() + 1) + int(codes_t[cb])
        
        print(f"  {t:4d}  {ms:4d}ms |", end="")
        for cb in range(n_cb):
            print(f" {int(codes_t[cb]):3d}", end="")
        print(f" | {combined:12d}")
    
    print("-" * (20 + n_cb * 5 + 18))
    print(f"\n{Colors.CYAN}Combined Token = (((CB0 × K + CB1) × K + CB2) × K + CB3) where K={int(codes_np.max())+1}{Colors.END}")


def display_code_diff(codes, frame1, frame2):
    """Show difference between two frames' codes."""
    codes_np = codes[0].cpu().numpy()
    T, n_cb = codes_np.shape
    
    print(f"\n{Colors.BOLD}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}  CODE DIFFERENCE - Frame {frame1} vs Frame {frame2}{Colors.END}")
    print(f"{Colors.BOLD}{'='*80}{Colors.END}")
    
    ms1 = frame1 * 10
    ms2 = frame2 * 10
    
    print(f"\n  Frame {frame1} ({ms1}ms): {codes_np[frame1]}")
    print(f"  Frame {frame2} ({ms2}ms): {codes_np[frame2]}")
    print(f"  Difference:              {codes_np[frame1] - codes_np[frame2]}")
    
    # Hamming-like distance (how many codebooks differ)
    diff = (codes_np[frame1] != codes_np[frame2]).sum()
    print(f"\n  Codebooks differing: {diff}/{n_cb}")
    
    if diff == 0:
        print(f"  {Colors.GREEN}→ Frames are IDENTICAL (same token){Colors.END}")
    elif diff == 1:
        print(f"  {Colors.YELLOW}→ Only 1 codebook differs (likely same phoneme, different detail){Colors.END}")
    else:
        print(f"  {Colors.RED}→ {diff} codebooks differ (different sounds){Colors.END}")


def export_codes_json(codes, output_path, metadata=None):
    """Export codes to a JSON file for external analysis."""
    codes_np = codes[0].cpu().numpy()
    T, n_cb = codes_np.shape
    
    export_data = {
        'metadata': metadata or {},
        'total_frames': T,
        'num_codebooks': n_cb,
        'codebook_size': int(codes_np.max()) + 1,
        'frame_rate_hz': 100,
        'frame_duration_ms': 10,
        'codes': codes_np.tolist(),
        'codebook_stats': {}
    }
    
    for cb in range(n_cb):
        counts = np.bincount(codes_np[:, cb], minlength=export_data['codebook_size'])
        unique = (counts > 0).sum()
        export_data['codebook_stats'][f'cb_{cb}'] = {
            'unique_codes': int(unique),
            'usage_percent': float(unique / export_data['codebook_size'] * 100),
            'top_10_codes': np.argsort(counts)[-10:][::-1].tolist(),
            'frequency': counts.tolist()
        }
    
    with open(output_path, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"\n{Colors.GREEN}✅ Codes exported to: {output_path}{Colors.END}")
    print(f"   Shape: {T} frames × {n_cb} codebooks")
    print(f"   File size: {Path(output_path).stat().st_size / 1024:.1f} KB")


def main():
    parser = argparse.ArgumentParser(
        description="Lipika Code Viewer - See your audio as discrete tokens",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View codes for a single file
  python view_codes.py --checkpoint checkpoints/ckpt_step00000201.pt --audio speech.wav
  
  # View with language specified
  python view_codes.py --checkpoint checkpoints/ckpt_step00000201.pt --audio speech.wav --lang hi
  
  # Export codes to JSON
  python view_codes.py --checkpoint checkpoints/ckpt_step00000201.pt --audio speech.wav --export codes.json
  
  # Compare two frames
  python view_codes.py --checkpoint checkpoints/ckpt_step00000201.pt --audio speech.wav --compare 10 50
  
  # Process all files in a directory
  python view_codes.py --checkpoint checkpoints/ckpt_step00000201.pt --audio-dir ./data --lang hi
        """
    )
    
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--audio", help="Path to audio file")
    parser.add_argument("--audio-dir", help="Path to directory of audio files")
    parser.add_argument("--lang", default="hi", help="Language code (default: hi)")
    parser.add_argument("--max-frames", type=int, default=60, help="Max frames to display")
    parser.add_argument("--export", help="Export codes to JSON file")
    parser.add_argument("--compare", nargs=2, type=int, help="Compare two frames (e.g., --compare 10 50)")
    parser.add_argument("--sequence-start", type=int, default=0, help="Start frame for sequence view")
    parser.add_argument("--device", default="auto")
    
    args = parser.parse_args()
    
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}  🎵 LIPIKA CODE VIEWER - See Your Audio Tokens 🎵{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*80}{Colors.END}")
    
    # Load model
    print(f"\n{Colors.CYAN}📦 Loading model...{Colors.END}")
    model = _load_model_from_checkpoint(args.checkpoint, args.device)
    print(f"   Model: {model.num_parameters()/1e6:.2f}M params")
    print(f"   Frame rate: {model.frame_rate:.1f} Hz")
    print(f"   Codebooks: {model.rvq.n_codebooks}")
    print(f"   Codebook size: {model.rvq_cfg.codebook_size}")
    
    # Determine audio files to process
    audio_files = []
    if args.audio:
        audio_files = [Path(args.audio)]
    elif args.audio_dir:
        audio_dir = Path(args.audio_dir)
        for ext in ['.wav', '.flac', '.mp3', '.ogg']:
            audio_files.extend(audio_dir.rglob(f"*{ext}"))
        audio_files = sorted(audio_files)
    
    if not audio_files:
        print(f"{Colors.RED}❌ No audio files specified!{Colors.END}")
        sys.exit(1)
    
    # Process each file
    for audio_path in audio_files:
        print(f"\n{Colors.BOLD}{'='*80}{Colors.END}")
        print(f"{Colors.BOLD}  📁 FILE: {audio_path.name}{Colors.END}")
        print(f"{Colors.BOLD}{'='*80}{Colors.END}")
        
        # Load audio
        audio, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)  # Stereo → mono
        
        if sr != 24000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=24000)
        
        duration = len(audio) / 24000
        print(f"\n{Colors.CYAN}📊 Audio Info:{Colors.END}")
        print(f"   Duration: {duration:.2f}s")
        print(f"   Samples: {len(audio)}")
        print(f"   Expected frames: {len(audio) // 240}")
        
        # Encode
        waveform = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0).to(model.encoder.stem.conv.weight.device)
        
        script_id = torch.tensor(
            [int(LANG_TO_SCRIPT.get(args.lang, ScriptFamily.DEVANAGARI))]
        ).to(waveform.device)
        
        with torch.no_grad():
            codes = model.encode(waveform, script_id)
        
        T_frames = codes.shape[1]
        print(f"   Actual frames: {T_frames}")
        print(f"   Codes shape: {tuple(codes.shape)}")
        print(f"   Total tokens: {codes.numel()}")
        print(f"   Compression: {len(audio) / codes.numel():.1f}x")
        
        # Codebook names
        codebook_names = [
            "CB0 (Semantic/Phoneme)",
            "CB1 (Articulation)",
            "CB2 (Prosody/Tone)",
            "CB3 (Timbre/Voice)"
        ][:model.rvq.n_codebooks]
        
        # 1. Display grid
        display_codes_grid(codes, codebook_names, args.max_frames)
        
        # 2. Statistics
        display_codebook_stats(codes, codebook_names)
        
        # 3. Sequence view
        display_code_sequence(codes, args.sequence_start, 30)
        
        # 4. Frame comparison (if requested)
        if args.compare:
            f1, f2 = args.compare
            if f1 < T_frames and f2 < T_frames:
                display_code_diff(codes, f1, f2)
        
        # 5. Export (if requested)
        if args.export:
            export_name = args.export
            if len(audio_files) > 1:
                export_name = f"{audio_path.stem}_{args.export}"
            export_codes_json(codes, export_name, {
                'file': str(audio_path),
                'duration_seconds': duration,
                'language': args.lang,
                'checkpoint': str(args.checkpoint)
            })
        
        # 6. Code histogram visualization (text-based)
        print(f"\n{Colors.BOLD}{'='*80}{Colors.END}")
        print(f"{Colors.BOLD}  CODE USAGE HISTOGRAM (text-based){Colors.END}")
        print(f"{Colors.BOLD}{'='*80}{Colors.END}")
        
        codes_np = codes[0].cpu().numpy()
        for cb in range(model.rvq.n_codebooks):
            counts = np.bincount(codes_np[:, cb], minlength=int(codes_np.max())+1)
            max_count = counts.max()
            
            print(f"\n{Colors.BOLD}{codebook_names[cb]}:{Colors.END}")
            # Show top 15 codes
            top_indices = np.argsort(counts)[-15:][::-1]
            for idx in top_indices:
                if counts[idx] > 0:
                    bar_len = int(40 * counts[idx] / max_count)
                    bar = "█" * bar_len
                    print(f"  Code {idx:4d}: {bar} {counts[idx]:4d}")
    
    print(f"\n{Colors.GREEN}{'='*80}{Colors.END}")
    print(f"{Colors.GREEN}  ✅ Done! Processed {len(audio_files)} file(s){Colors.END}")
    print(f"{Colors.GREEN}{'='*80}{Colors.END}")


if __name__ == "__main__":
    main()