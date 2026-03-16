#!/usr/bin/env python3
"""
LIPIKA TOKENIZER - Convert Audio to Discrete Tokens
============================================================
Fixed version - handles dimension mismatches gracefully
============================================================
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
import argparse
from datetime import datetime

from audio import AudioPreprocessor
from encoder import create_audio_encoder

# Try to import from lipika, but we'll handle dimension mismatches
try:
    from lipika import ResidualVectorQuantizer as LipikaRVQ
    from lipika import RVQConfig, ModelConfig
    LIPIKA_AVAILABLE = True
    print("✓ Found Lipika's RVQ implementation")
except ImportError:
    LIPIKA_AVAILABLE = False
    print("Using built-in RVQ implementation")


# =============================================================================
# SIMPLE VECTOR QUANTIZER (Works with any dimensions)
# =============================================================================

class SimpleVectorQuantizer(nn.Module):
    """Simple EMA Vector Quantizer with flexible dimensions."""
    
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # Initialize codebook
        self.embedding = nn.Parameter(torch.randn(num_embeddings, embedding_dim))
        self.register_buffer("cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("embedding_avg", self.embedding.data.clone())
        
        # Initialize
        nn.init.normal_(self.embedding, mean=0.0, std=1.0 / embedding_dim**0.5)
        
    def forward(self, z):
        """
        Args:
            z: (batch, frames, dim)
        
        Returns:
            z_q: quantized vectors (same shape as input)
            indices: token indices (batch, frames)
            loss: commitment loss
        """
        batch_size, frames, dim = z.shape
        flat_z = z.reshape(-1, dim)
        
        # Calculate distances
        distances = (flat_z.pow(2).sum(1, keepdim=True) 
                    - 2 * flat_z @ self.embedding.t() 
                    + self.embedding.pow(2).sum(1))
        
        # Encoding
        indices = distances.argmin(1)
        z_q = self.embedding[indices].view(batch_size, frames, dim)
        
        # Loss
        commitment_loss = F.mse_loss(z_q.detach(), z)
        vq_loss = commitment_loss * self.commitment_cost
        
        # Straight-through estimator
        z_q_st = z + (z_q - z).detach()
        
        return z_q_st, indices.view(batch_size, frames), vq_loss


class SimpleResidualVQ(nn.Module):
    """
    Simple Residual Vector Quantizer - Works with any input dimension.
    """
    
    def __init__(
        self,
        n_codebooks: int = 8,
        codebook_size: int = 1024,
        codebook_dim: int = 128,
        input_dim: int = 768
    ):
        super().__init__()
        
        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.input_dim = input_dim
        
        print(f"  Creating RVQ: input_dim={input_dim} → codebook_dim={codebook_dim}")
        
        # Project input to codebook dimension
        self.input_proj = nn.Linear(input_dim, codebook_dim)
        
        # Create codebooks
        self.codebooks = nn.ModuleList([
            SimpleVectorQuantizer(codebook_size, codebook_dim)
            for _ in range(n_codebooks)
        ])
        
        # Project back to input dimension
        self.output_proj = nn.Linear(codebook_dim, input_dim)
        
    def forward(self, z):
        """
        Args:
            z: (batch, frames, input_dim)
        
        Returns:
            codes: (batch, frames, n_codebooks) token indices
            loss: total VQ loss
        """
        # Project to codebook dimension
        z_proj = self.input_proj(z)
        
        # Residual quantization
        residual = z_proj
        all_indices = []
        total_loss = 0
        
        for i, vq in enumerate(self.codebooks):
            z_q, indices, loss = vq(residual)
            residual = residual - z_q.detach()
            all_indices.append(indices)
            total_loss += loss
        
        # Stack indices
        codes = torch.stack(all_indices, dim=-1)  # (batch, frames, n_codebooks)
        
        return codes, total_loss
    
    @torch.no_grad()
    def encode(self, z):
        """Encode to discrete codes only."""
        z_proj = self.input_proj(z)
        residual = z_proj
        all_indices = []
        
        for vq in self.codebooks:
            z_q, indices, _ = vq(residual)
            residual = residual - z_q
            all_indices.append(indices)
        
        return torch.stack(all_indices, dim=-1)
    
    @torch.no_grad()
    def decode(self, codes):
        """Decode from discrete codes."""
        z_q = torch.zeros(
            codes.shape[0], codes.shape[1], self.codebook_dim,
            device=codes.device
        )
        
        for i, vq in enumerate(self.codebooks):
            indices = codes[..., i]
            z_q_i = vq.embedding[indices]
            z_q = z_q + z_q_i
        
        return self.output_proj(z_q)


# =============================================================================
# ADAPTER FOR LIPIKA'S RVQ (if dimensions don't match)
# =============================================================================

class LipikaRVQAdapter(nn.Module):
    """
    Adapter to make Lipika's RVQ work with our dimensions.
    Lipika expects encoder_channels=512, but we have 768-dim embeddings.
    """
    
    def __init__(self, n_codebooks: int = 8, codebook_size: int = 1024):
        super().__init__()
        
        # Project from 768 to 512 (Lipika's expected dim)
        self.proj_in = nn.Linear(768, 512)
        
        # Use Lipika's RVQ
        from lipika import ResidualVectorQuantizer as LipikaRVQ
        from lipika import RVQConfig, ModelConfig
        
        rvq_config = RVQConfig(
            n_codebooks=n_codebooks,
            codebook_size=codebook_size,
            codebook_dim=128
        )
        model_config = ModelConfig(encoder_channels=512)
        self.rvq = LipikaRVQ(rvq_config, model_config)
        
        # Project back to 768
        self.proj_out = nn.Linear(512, 768)
        
        print(f"  Created Lipika adapter: 768 → 512 → RVQ → 512 → 768")
        
    def forward(self, z):
        """
        Args:
            z: (batch, frames, 768)
        
        Returns:
            codes: (batch, frames, n_codebooks)
            loss: VQ loss
        """
        # Project to Lipika's dimension
        z_proj = self.proj_in(z)
        
        # Run through Lipika's RVQ
        # Lipika's RVQ returns a dict with 'codes' and 'vq_loss'
        output = self.rvq(z_proj, w2v_targets=None)
        
        codes = output['codes']
        loss = output['vq_loss']
        
        return codes, loss


# =============================================================================
# MAIN TOKENIZER CLASS
# =============================================================================

class AudioDiscreteTokenizer:
    """
    Convert audio to discrete tokens using Vector Quantization.
    Handles dimension mismatches automatically.
    """
    
    def __init__(
        self,
        n_codebooks: int = 8,
        codebook_size: int = 1024,
        device: str = "auto"
    ):
        print("\n" + "="*70)
        print("INITIALIZING AUDIO DISCRETE TOKENIZER")
        print("="*70)
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Device: {self.device}")
        print(f"Configuration: {n_codebooks} codebooks × {codebook_size} tokens")
        
        # 1. AudioPreprocessor
        print("\n[1/4] Loading AudioPreprocessor...")
        self.preprocessor = AudioPreprocessor(
            target_sr=24000,
            peak_norm=True
        )
        
        # 2. AudioEncoder
        print("[2/4] Loading AudioEncoder (base)...")
        self.encoder = create_audio_encoder(
            model_size="base",
            sample_rate=24000,
            window_seconds=5.0
        )
        self.encoder = self.encoder.to(self.device)
        self.encoder.eval()
        
        # 3. Residual Vector Quantizer
        print("[3/4] Initializing Residual Vector Quantizer...")
        
        # Try different RVQ implementations
        self.rvq = self._create_rvq(n_codebooks, codebook_size)
        
        # 4. Initialize codebooks
        print("[4/4] Initializing codebooks...")
        self._init_codebooks()
        
        # Statistics
        self.files_processed = 0
        self.total_tokens = 0
        
        print("\n" + "="*70)
        print("✓ TOKENIZER READY!")
        print(f"  Output shape: (batch, frames, {n_codebooks})")
        print(f"  Frames for 5s: {self._get_frame_count()}")
        print("="*70 + "\n")
    
    def _get_frame_count(self) -> int:
        """Calculate expected frame count for 5 seconds."""
        return 1501  # From your validation report
    
    def _create_rvq(self, n_codebooks: int, codebook_size: int):
        """Create RVQ with fallback options."""
        
        # Option 1: Try Lipika with adapter
        if LIPIKA_AVAILABLE:
            try:
                print("  Attempting to use Lipika's RVQ with adapter...")
                rvq = LipikaRVQAdapter(n_codebooks, codebook_size)
                print("  ✓ Using Lipika's RVQ (with adapter)")
                return rvq
            except Exception as e:
                print(f"  ⚠ Lipika adapter failed: {e}")
        
        # Option 2: Use Simple RVQ
        print("  Using simple RVQ implementation")
        return SimpleResidualVQ(
            n_codebooks=n_codebooks,
            codebook_size=codebook_size,
            codebook_dim=128,
            input_dim=768
        )
    
    def _init_codebooks(self):
        """Initialize codebooks."""
        with torch.no_grad():
            # Handle different RVQ structures
            if hasattr(self.rvq, 'codebooks'):
                for vq in self.rvq.codebooks:
                    if hasattr(vq, 'embedding'):
                        vq.embedding.data = F.normalize(vq.embedding.data, dim=1)
            elif hasattr(self.rvq, 'rvq') and hasattr(self.rvq.rvq, 'codebooks'):
                # Lipika adapter case
                for vq in self.rvq.rvq.codebooks:
                    if hasattr(vq, 'embedding'):
                        vq.embedding.data = F.normalize(vq.embedding.data, dim=1)
    
    @torch.no_grad()
    def tokenize_file(self, audio_path: str, max_duration: float = 5.0) -> dict:
        """
        Tokenize an audio file to discrete tokens.
        """
        print(f"\n📂 Processing: {os.path.basename(audio_path)}")
        start_time = datetime.now()
        
        # Step 1: Preprocess audio
        print("  ⚙️ Preprocessing...")
        waveform = self.preprocessor.process(
            audio_path,
            max_duration=max_duration
        )
        waveform = waveform.to(self.device)
        
        # Step 2: Encode to embeddings
        print("  ⚙️ Encoding to embeddings...")
        output = self.encoder(waveform, return_pooled=False)
        embeddings = output if isinstance(output, torch.Tensor) else output.embeddings
        
        print(f"  Embeddings shape: {tuple(embeddings.shape)}")
        
        # Step 3: Quantize to discrete tokens
        print("  ⚙️ Quantizing to discrete tokens...")
        tokens, vq_loss = self.rvq(embeddings)
        
        print(f"  Tokens shape: {tuple(tokens.shape)}")
        
        # Step 4: Collect metadata
        info = self.preprocessor.inspect(audio_path)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        metadata = {
            'file': audio_path,
            'file_size': os.path.getsize(audio_path),
            'original_duration': info['duration_s'],
            'original_sample_rate': info['original_sr'],
            'processed_duration': max_duration,
            'processed_sample_rate': 24000,
            'frames': tokens.shape[1],
            'codebooks': tokens.shape[2],
            'codebook_size': 1024,
            'total_tokens': tokens.numel(),
            'vq_loss': float(vq_loss.item()) if hasattr(vq_loss, 'item') else float(vq_loss),
            'processing_time': processing_time,
            'timestamp': datetime.now().isoformat(),
            'device': str(self.device)
        }
        
        # Update statistics
        self.files_processed += 1
        self.total_tokens += tokens.numel()
        
        print(f"  ✅ Generated {tokens.shape[1]} frames × {tokens.shape[2]} codebooks = {tokens.numel():,} tokens")
        print(f"  ⏱️  Time: {processing_time:.2f}s")
        
        return {
            'tokens': tokens.cpu(),
            'embeddings': embeddings.cpu(),
            'vq_loss': vq_loss.cpu() if hasattr(vq_loss, 'cpu') else torch.tensor(vq_loss),
            'metadata': metadata
        }
    
    def save_tokens(self, result: dict, output_dir: str = "./tokens"):
        """Save tokens in multiple formats."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        base_name = Path(result['metadata']['file']).stem
        
        # Save as PyTorch tensor
        torch.save({
            'tokens': result['tokens'],
            'embeddings': result['embeddings'],
            'metadata': result['metadata']
        }, output_dir / f"{base_name}_tokens.pt")
        print(f"  💾 Saved: {output_dir}/{base_name}_tokens.pt")
        
        # Save as NumPy
        np.save(output_dir / f"{base_name}_tokens.npy", result['tokens'].numpy())
        print(f"  💾 Saved: {output_dir}/{base_name}_tokens.npy")
        
        # Save as CSV (first 100 frames)
        self._save_as_csv(result['tokens'], output_dir / f"{base_name}_tokens.csv")
        print(f"  💾 Saved: {output_dir}/{base_name}_tokens.csv")
        
        # Save metadata
        with open(output_dir / f"{base_name}_metadata.json", 'w', encoding='utf-8') as f:
            json.dump(result['metadata'], f, indent=2)
        print(f"  💾 Saved: {output_dir}/{base_name}_metadata.json")
    
    def _save_as_csv(self, tokens: torch.Tensor, path: Path):
        """Save tokens as CSV for human readability."""
        tokens = tokens.squeeze(0)  # (frames, codebooks)
        
        lines = ["frame," + ",".join(f"cb{i}" for i in range(tokens.shape[1]))]
        for f in range(min(tokens.shape[0], 100)):  # First 100 frames
            lines.append(f"{f}," + ",".join(str(tokens[f, c].item()) for c in range(tokens.shape[1])))
        
        if tokens.shape[0] > 100:
            lines.append("... (truncated, full tensor in .pt file)")
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
    
    def analyze_tokens(self, tokens: torch.Tensor) -> dict:
        """Analyze token distribution and codebook usage."""
        tokens_np = tokens.squeeze(0).numpy()  # (frames, codebooks)
        
        analysis = {
            'total_frames': tokens_np.shape[0],
            'total_tokens': tokens_np.size,
            'codebooks': []
        }
        
        for cb in range(tokens_np.shape[1]):
            codes = tokens_np[:, cb]
            unique, counts = np.unique(codes, return_counts=True)
            
            usage_pct = len(unique) / 1024 * 100
            entropy = -np.sum((counts/len(codes)) * np.log(counts/len(codes) + 1e-12))
            perplexity = np.exp(entropy)
            
            analysis['codebooks'].append({
                'unique_codes': len(unique),
                'usage_percent': round(usage_pct, 2),
                'perplexity': round(perplexity, 2),
                'most_common': int(unique[np.argmax(counts)]),
                'most_common_count': int(np.max(counts))
            })
        
        return analysis


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert audio to discrete tokens using Vector Quantization"
    )
    
    parser.add_argument('input', nargs='+', help='Input audio file(s)')
    parser.add_argument('--output-dir', '-o', default='./tokens',
                       help='Output directory (default: ./tokens)')
    parser.add_argument('--duration', '-d', type=float, default=5.0,
                       help='Max duration in seconds (default: 5.0)')
    parser.add_argument('--codebooks', '-c', type=int, default=8,
                       help='Number of codebooks (default: 8)')
    parser.add_argument('--codebook-size', '-s', type=int, default=1024,
                       help='Size of each codebook (default: 1024)')
    parser.add_argument('--device', default='auto',
                       help='Device: auto/cuda/cpu (default: auto)')
    parser.add_argument('--analyze', '-a', action='store_true',
                       help='Show token analysis')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Initialize tokenizer
    tokenizer = AudioDiscreteTokenizer(
        n_codebooks=args.codebooks,
        codebook_size=args.codebook_size,
        device=args.device
    )
    
    # Process files
    for audio_path in args.input:
        if not os.path.exists(audio_path):
            print(f"❌ File not found: {audio_path}")
            continue
        
        try:
            # Tokenize
            result = tokenizer.tokenize_file(audio_path, max_duration=args.duration)
            
            # Analyze if requested
            if args.analyze:
                analysis = tokenizer.analyze_tokens(result['tokens'])
                print("\n📊 TOKEN ANALYSIS")
                print("="*70)
                print(f"Total frames: {analysis['total_frames']}")
                print(f"Total tokens: {analysis['total_tokens']:,}")
                print("-"*70)
                print(f"{'Codebook':<10} {'Usage %':<10} {'Perplexity':<12} {'Most Common'}")
                print("-"*70)
                for i, cb in enumerate(analysis['codebooks']):
                    print(f"CB{i:<8} {cb['usage_percent']:<9}% {cb['perplexity']:<12} {cb['most_common']}")
                print("="*70)
            
            # Save tokens
            tokenizer.save_tokens(result, output_dir=args.output_dir)
            
            print(f"\n✅ Successfully tokenized: {audio_path}")
            
        except Exception as e:
            print(f"❌ Error processing {audio_path}: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    main()