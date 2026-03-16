#!/usr/bin/env python3
"""
===============================================================================
LIPIKA LEARNING LAB - Interactive Tutorial & Experimentation Suite
===============================================================================

This file provides an interactive way to explore all of Lipika's features:
- Loading and inspecting the model
- Training visualization
- Encoding/decoding audio
- Codebook analysis
- Discriminator behavior
- Loss function experiments

Run with: python lipika_lab.py
===============================================================================
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from collections import defaultdict

# Import the BEAST!
try:
    from lipika import (
        # Core models
        LipikaTokenizer,
        AudioEncoder,
        AudioDecoder,
        ResidualVectorQuantizer,
        VectorQuantizerEMA,
        
        # Discriminators
        MultiScaleMultiPeriodDiscriminator,
        PeriodDiscriminator,
        ScaleDiscriminator,
        
        # Loss functions
        MelSpectrogramLoss,
        MultiScaleSTFTLoss,
        hinge_disc_loss,
        hinge_gen_loss,
        feature_matching_loss,
        
        # Configs
        AudioConfig,
        RVQConfig,
        ModelConfig,
        TrainingConfig,
        
        # Utilities
        get_device,
        device_info,
        ScriptFamily,
        LANG_TO_SCRIPT,
        
        # Dataset
        AudioDataset,
        SyntheticAudioDataset,
        
        # Training
        train,
        validate,
        
        # Inference
        encode_audio_file,
        decode_codes_to_file,
        
        # Monitoring
        CodebookMonitor,
        MetricsTracker,
        
        # Plotting
        plot_training_curves,
        plot_spectrogram_comparison,
        
        # Checkpoint
        CheckpointManager,
        _load_model_from_checkpoint,
        
        # Device
        DEVICE
    )
    from audio import AudioPreprocessor
    from encoder import create_audio_encoder
    
    LIPIKA_AVAILABLE = True
    print("✅ Successfully imported Lipika - The BEAST is ready!")
    
except ImportError as e:
    print(f"❌ Failed to import Lipika: {e}")
    print("Make sure lipika.py is in the same directory")
    sys.exit(1)


class LipikaLearningLab:
    """
    Interactive learning environment for Lipika Tokenizer.
    
    This class provides methods to explore every aspect of Lipika:
    - Model architecture
    - Training dynamics
    - Codebook behavior
    - Discriminator responses
    - Loss landscapes
    """
    
    def __init__(self, device: str = "auto"):
        print("\n" + "="*80)
        print("🎓 LIPIKA LEARNING LAB - Interactive Tutorial")
        print("="*80)
        
        # Setup device
        self.device = get_device(device)
        print(f"\n📡 Device: {device_info(self.device)}")
        
        # Create experiment directory
        self.exp_dir = Path("./lipika_experiments")
        self.exp_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self._init_configs()
        self._init_models()
        self._init_discriminators()
        self._init_losses()
        
        # Storage for experiments
        self.results = {}
        
        print("\n" + "="*80)
        print("🎓 Lab ready! Use the methods below to explore:")
        self._print_menu()
        print("="*80)
    
    def _print_menu(self):
        """Print available experiments."""
        print("\n📋 AVAILABLE EXPERIMENTS:")
        experiments = [
            ("1. explore_model()", "Inspect model architecture"),
            ("2. visualize_codebooks()", "See codebook distributions"),
            ("3. test_quantization()", "Test VQ on random data"),
            ("4. watch_training()", "Monitor training dynamics"),
            ("5. analyze_discriminator()", "Test discriminator responses"),
            ("6. loss_landscape()", "Visualize loss functions"),
            ("7. encode_decode_demo()", "Full encoding/decoding pipeline"),
            ("8. codebook_health()", "Monitor codebook usage"),
            ("9. script_conditioning()", "Test script-family adapter"),
            ("10. batch_experiment()", "Test batch processing"),
            ("11. compare_presets()", "Compare model sizes"),
            ("12. profile_performance()", "Measure speed/memory"),
        ]
        for exp, desc in experiments:
            print(f"   {exp:25} - {desc}")
    
    def _init_configs(self):
        """Initialize all configurations."""
        print("\n📋 Initializing configurations...")
        
        # Audio config
        self.audio_cfg = AudioConfig(
            sample_rate=24000,
            n_fft=2048,
            hop_length=240,
            n_mels=128
        )
        print(f"  ✓ AudioConfig: {self.audio_cfg.sample_rate}Hz, {self.audio_cfg.n_mels} mel bands")
        
        # RVQ config (small for experiments)
        self.rvq_cfg = RVQConfig(
            n_codebooks=4,           # Smaller for faster experiments
            codebook_size=256,        # Smaller codebook
            codebook_dim=64,
            commitment_cost=1.0,
            ema_decay=0.99
        )
        print(f"  ✓ RVQConfig: {self.rvq_cfg.n_codebooks} codebooks × {self.rvq_cfg.codebook_size}")
        
        # Model config (small for experiments)
        self.model_cfg = ModelConfig(
            encoder_channels=128,
            decoder_channels=128,
            disc_channels=32,
            disc_depth=2,
            n_script_families=12
        )
        print(f"  ✓ ModelConfig: {self.model_cfg.encoder_channels} channels")
        
        # Training config
        self.train_cfg = TrainingConfig(
            batch_size=2,
            num_epochs=1,
            max_duration=1.0,
            device=str(self.device)
        )
        print(f"  ✓ TrainingConfig: batch={self.train_cfg.batch_size}")
    
    def _init_models(self):
        """Initialize all models."""
        print("\n🤖 Initializing models...")
        
        # Main Lipika model
        self.model = LipikaTokenizer(
            self.audio_cfg,
            self.rvq_cfg,
            self.model_cfg,
            use_semantic_teacher=False  # Disable for experiments
        ).to(self.device)
        print(f"  ✓ LipikaTokenizer: {self.model.num_parameters()/1e6:.2f}M params")
        
        # Individual components (for detailed study)
        self.encoder = self.model.encoder
        self.rvq = self.model.rvq
        self.decoder = self.model.decoder
        self.script_adapter = self.model.script_adapter
        
        print(f"  ✓ Encoder: {sum(p.numel() for p in self.encoder.parameters())/1e3:.0f}K params")
        print(f"  ✓ RVQ: {sum(p.numel() for p in self.rvq.parameters())/1e3:.0f}K params")
        print(f"  ✓ Decoder: {sum(p.numel() for p in self.decoder.parameters())/1e3:.0f}K params")
    
    def _init_discriminators(self):
        """Initialize discriminators."""
        print("\n⚔️ Initializing discriminators...")
        
        self.discriminator = MultiScaleMultiPeriodDiscriminator(self.model_cfg).to(self.device)
        print(f"  ✓ MultiScaleMultiPeriodDiscriminator")
        print(f"    - MSDs: {len(self.discriminator.msds)} scales")
        print(f"    - MPDs: {len(self.discriminator.mpds)} periods")
    
    def _init_losses(self):
        """Initialize loss functions."""
        print("\n📉 Initializing loss functions...")
        
        self.mel_loss = MelSpectrogramLoss(self.audio_cfg)
        self.stft_loss = MultiScaleSTFTLoss()
        
        print(f"  ✓ MelSpectrogramLoss")
        print(f"  ✓ MultiScaleSTFTLoss")
        print(f"  ✓ Hinge losses ready")
        print(f"  ✓ Feature matching ready")
    
    # =========================================================================
    # EXPERIMENT 1: Model Architecture Exploration
    # =========================================================================
    
    def explore_model(self):
        """Print detailed model architecture."""
        print("\n" + "="*80)
        print("🔍 EXPERIMENT 1: Model Architecture Exploration")
        print("="*80)
        
        def print_module(name, module, indent=0):
            spaces = "  " * indent
            print(f"{spaces}📦 {name}: {module.__class__.__name__}")
            if hasattr(module, 'parameters'):
                params = sum(p.numel() for p in module.parameters())
                print(f"{spaces}   Parameters: {params:,}")
            if hasattr(module, 'training'):
                print(f"{spaces}   Mode: {'train' if module.training else 'eval'}")
        
        print("\n🏗️ ENCODER ARCHITECTURE:")
        print_module("Encoder", self.encoder)
        print(f"   Compression ratio: {self.encoder.compression_ratio}x")
        print(f"   Frame rate: {self.model.frame_rate:.1f} Hz")
        
        print("\n📚 RVQ ARCHITECTURE:")
        print_module("RVQ", self.rvq)
        print(f"   Codebooks: {self.rvq.n_codebooks}")
        print(f"   Codebook dim: {self.rvq.codebook_dim}")
        
        print("\n🎚️ SCRIPT ADAPTER:")
        print_module("ScriptAdapter", self.script_adapter)
        print(f"   Script families: {self.model_cfg.n_script_families}")
        print(f"   Retroflex scripts: {self.script_adapter.RETROFLEX_SCRIPTS}")
        
        print("\n🔊 DECODER ARCHITECTURE:")
        print_module("Decoder", self.decoder)
        
        return {
            'encoder_params': sum(p.numel() for p in self.encoder.parameters()),
            'rvq_params': sum(p.numel() for p in self.rvq.parameters()),
            'decoder_params': sum(p.numel() for p in self.decoder.parameters()),
            'total_params': self.model.num_parameters(),
            'frame_rate': self.model.frame_rate
        }
    
    # =========================================================================
    # EXPERIMENT 2: Codebook Visualization
    # =========================================================================
    
    def visualize_codebooks(self, n_samples: int = 1000):
        """Visualize codebook distributions."""
        print("\n" + "="*80)
        print("📊 EXPERIMENT 2: Codebook Visualization")
        print("="*80)
        
        # Generate random embeddings
        z = torch.randn(1, n_samples, self.model_cfg.encoder_channels).to(self.device)
        
        # Quantize
        with torch.no_grad():
            output = self.rvq(z)
            codes = output['codes']  # (1, n_samples, n_codebooks)
        
        codes_np = codes[0].cpu().numpy()
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Codebook Usage Analysis", fontsize=16)
        
        for cb in range(min(4, self.rvq.n_codebooks)):
            ax = axes[cb//2, cb%2]
            hist = np.bincount(codes_np[:, cb], minlength=self.rvq_cfg.codebook_size)
            ax.bar(range(self.rvq_cfg.codebook_size), hist, alpha=0.7)
            ax.set_title(f"Codebook {cb} - Usage Distribution")
            ax.set_xlabel("Token ID")
            ax.set_ylabel("Frequency")
            ax.set_xlim(0, self.rvq_cfg.codebook_size)
            
            # Stats
            unique = np.sum(hist > 0)
            usage_pct = unique / self.rvq_cfg.codebook_size * 100
            ax.text(0.02, 0.98, f"Usage: {usage_pct:.1f}%", 
                   transform=ax.transAxes, va='top')
        
        plt.tight_layout()
        plt.savefig(self.exp_dir / "codebook_analysis.png")
        plt.show()
        
        print(f"\n📈 Codebook usage statistics:")
        for cb in range(self.rvq.n_codebooks):
            unique = len(np.unique(codes_np[:, cb]))
            usage = unique / self.rvq_cfg.codebook_size * 100
            print(f"  Codebook {cb}: {unique:4d}/{self.rvq_cfg.codebook_size} tokens used ({usage:.1f}%)")
        
        # Calculate perplexity
        probs = []
        for cb in range(self.rvq.n_codebooks):
            counts = np.bincount(codes_np[:, cb], minlength=self.rvq_cfg.codebook_size)
            probs_cb = counts / counts.sum()
            probs_cb = probs_cb[probs_cb > 0]
            entropy = -np.sum(probs_cb * np.log(probs_cb))
            perplexity = np.exp(entropy)
            print(f"  Codebook {cb} perplexity: {perplexity:.2f} (max={self.rvq_cfg.codebook_size})")
        
        return {
            'codes': codes_np,
            'usage': [len(np.unique(codes_np[:, cb])) for cb in range(self.rvq.n_codebooks)]
        }
    
    # =========================================================================
    # EXPERIMENT 3: Quantization Test
    # =========================================================================
    
    def test_quantization(self, n_tests: int = 10):
        """Test quantization error on random inputs."""
        print("\n" + "="*80)
        print("🔢 EXPERIMENT 3: Quantization Error Analysis")
        print("="*80)
        
        errors = []
        
        for i in range(n_tests):
            # Create random input
            z = torch.randn(1, 100, self.model_cfg.encoder_channels).to(self.device)
            
            # Quantize and reconstruct
            with torch.no_grad():
                codes, _ = self.rvq(z)  # Get codes
                z_q = self.rvq.decode(codes)  # Reconstruct
            
            # Calculate error
            mse = F.mse_loss(z_q, z).item()
            errors.append(mse)
            print(f"  Test {i+1:2d}: MSE = {mse:.6f}")
        
        avg_error = np.mean(errors)
        std_error = np.std(errors)
        
        print(f"\n📊 Summary:")
        print(f"  Mean MSE: {avg_error:.6f}")
        print(f"  Std MSE:  {std_error:.6f}")
        print(f"  SNR:      {20 * np.log10(1.0/avg_error):.2f} dB")
        
        return {'errors': errors, 'avg_error': avg_error}
    
    # =========================================================================
    # EXPERIMENT 4: Training Dynamics Monitor
    # =========================================================================
    
    def watch_training(self, n_steps: int = 100):
        """Monitor training dynamics with synthetic data."""
        print("\n" + "="*80)
        print("📈 EXPERIMENT 4: Training Dynamics Monitor")
        print("="*80)
        
        # Create synthetic dataset
        dataset = SyntheticAudioDataset(self.audio_cfg, n_samples=50, max_duration=1.0)
        loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
        
        # Setup optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        
        # Track metrics
        losses = []
        grad_norms = []
        
        self.model.train()
        
        print("\n🚀 Starting training simulation...")
        for step, batch in enumerate(loader):
            if step >= n_steps:
                break
            
            waveform = batch['waveform'].to(self.device)
            script_ids = batch['script_id'].to(self.device)
            
            # Forward pass
            output = self.model(waveform, script_ids)
            
            # Calculate loss
            loss = (
                output['recon_loss'] +
                output['mel_loss'] * 0.1 +
                output['stft_loss'] +
                output['vq_loss']
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Track gradient norm
            total_norm = 0
            for p in self.model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.norm().item() ** 2
            grad_norm = total_norm ** 0.5
            
            optimizer.step()
            
            losses.append(loss.item())
            grad_norms.append(grad_norm)
            
            if step % 10 == 0:
                print(f"  Step {step:3d}: loss={loss.item():.4f}, grad_norm={grad_norm:.4f}")
        
        # Plot training dynamics
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(losses)
        ax1.set_title("Training Loss")
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Loss")
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(grad_norms)
        ax2.set_title("Gradient Norm")
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Norm")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.exp_dir / "training_dynamics.png")
        plt.show()
        
        print(f"\n📊 Summary:")
        print(f"  Final loss: {losses[-1]:.4f}")
        print(f"  Loss std:   {np.std(losses):.4f}")
        print(f"  Avg grad norm: {np.mean(grad_norms):.4f}")
        
        return {'losses': losses, 'grad_norms': grad_norms}
    
    # =========================================================================
    # EXPERIMENT 5: Discriminator Analysis
    # =========================================================================
    
    def analyze_discriminator(self):
        """Test discriminator responses to real/fake audio."""
        print("\n" + "="*80)
        print("⚔️ EXPERIMENT 5: Discriminator Analysis")
        print("="*80)
        
        # Create synthetic "real" and "fake" audio
        real_audio = torch.randn(2, 1, 24000).to(self.device) * 0.5
        fake_audio = torch.randn(2, 1, 24000).to(self.device) * 0.3 + 0.1
        
        # Get discriminator outputs
        with torch.no_grad():
            real_logits, real_feats = self.discriminator(real_audio)
            fake_logits, fake_feats = self.discriminator(fake_audio)
        
        print(f"\n📊 Discriminator outputs:")
        print(f"  MSD scales: {len(self.discriminator.msds)}")
        print(f"  MPD periods: {len(self.discriminator.mpds)}")
        print(f"  Total outputs: {len(real_logits)}")
        
        print("\n📈 Logits analysis:")
        for i, (real, fake) in enumerate(zip(real_logits, fake_logits)):
            real_mean = real.mean().item()
            fake_mean = fake.mean().item()
            separation = real_mean - fake_mean
            print(f"  Output {i:2d}: real={real_mean:+.3f}, fake={fake_mean:+.3f}, sep={separation:+.3f}")
        
        # Calculate hinge losses
        d_loss = hinge_disc_loss(real_logits, fake_logits)
        g_loss = hinge_gen_loss(fake_logits)
        fm_loss = feature_matching_loss(real_feats, fake_feats)
        
        print(f"\n🎯 Loss values:")
        print(f"  Discriminator loss: {d_loss.item():.4f}")
        print(f"  Generator loss:     {g_loss.item():.4f}")
        print(f"  Feature matching:   {fm_loss.item():.4f}")
        
        return {
            'real_logits': [l.mean().item() for l in real_logits],
            'fake_logits': [l.mean().item() for l in fake_logits],
            'd_loss': d_loss.item(),
            'g_loss': g_loss.item(),
            'fm_loss': fm_loss.item()
        }
    
    # =========================================================================
    # EXPERIMENT 6: Loss Landscape
    # =========================================================================
    
    def loss_landscape(self, resolution: int = 20):
        """Visualize loss landscape around a point."""
        print("\n" + "="*80)
        print("🗺️ EXPERIMENT 6: Loss Landscape Visualization")
        print("="*80)
        
        # Create reference point
        z = torch.randn(1, 50, self.model_cfg.encoder_channels).to(self.device)
        z.requires_grad_(True)
        
        # Create two random directions
        dir1 = torch.randn_like(z)
        dir2 = torch.randn_like(z)
        dir1 = dir1 / dir1.norm()
        dir2 = dir2 / dir2.norm()
        
        # Sample loss on grid
        alpha = np.linspace(-2, 2, resolution)
        beta = np.linspace(-2, 2, resolution)
        loss_grid = np.zeros((resolution, resolution))
        
        print(f"\n🔍 Sampling {resolution}x{resolution} grid...")
        for i, a in enumerate(alpha):
            for j, b in enumerate(beta):
                z_perturbed = z + a * dir1 + b * dir2
                with torch.no_grad():
                    codes, _ = self.rvq(z_perturbed)
                    z_q = self.rvq.decode(codes)
                    loss = F.mse_loss(z_q, z_perturbed).item()
                loss_grid[i, j] = loss
        
        # Plot landscape
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Contour plot
        X, Y = np.meshgrid(alpha, beta)
        contour = ax1.contourf(X, Y, loss_grid.T, levels=20, cmap='viridis')
        ax1.set_xlabel("Direction 1")
        ax1.set_ylabel("Direction 2")
        ax1.set_title("Loss Contour")
        plt.colorbar(contour, ax=ax1)
        
        # 3D surface plot
        ax2 = fig.add_subplot(122, projection='3d')
        surf = ax2.plot_surface(X, Y, loss_grid.T, cmap='viridis', alpha=0.8)
        ax2.set_xlabel("Dir1")
        ax2.set_ylabel("Dir2")
        ax2.set_zlabel("Loss")
        ax2.set_title("Loss Surface")
        
        plt.tight_layout()
        plt.savefig(self.exp_dir / "loss_landscape.png")
        plt.show()
        
        print(f"\n📊 Loss statistics:")
        print(f"  Min loss: {loss_grid.min():.6f}")
        print(f"  Max loss: {loss_grid.max():.6f}")
        print(f"  Mean loss: {loss_grid.mean():.6f}")
        print(f"  Std loss: {loss_grid.std():.6f}")
        
        return {'loss_grid': loss_grid, 'alpha': alpha, 'beta': beta}
    
    # =========================================================================
    # EXPERIMENT 7: Full Pipeline Demo
    # =========================================================================
    
    def encode_decode_demo(self, duration: float = 1.0):
        """Demonstrate full encode-decode pipeline."""
        print("\n" + "="*80)
        print("🔄 EXPERIMENT 7: Full Encode-Decode Pipeline")
        print("="*80)
        
        # Create synthetic audio
        print(f"\n🎵 Creating {duration}s synthetic audio...")
        samples = int(duration * self.audio_cfg.sample_rate)
        t = torch.linspace(0, duration, samples)
        waveform = (0.5 * torch.sin(2 * np.pi * 440 * t) +  # 440 Hz tone
                   0.25 * torch.sin(2 * np.pi * 880 * t))   # 880 Hz harmonic
        waveform = waveform.unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,T)
        
        # Random script ID (Hindi)
        script_id = torch.tensor([ScriptFamily.DEVANAGARI]).to(self.device)
        
        print("\n📈 Pipeline stages:")
        
        # 1. Encode to codes
        print("  Step 1: Encoding to codes...")
        start = time.time()
        with torch.no_grad():
            codes = self.model.encode(waveform, script_id)
        encode_time = time.time() - start
        print(f"    ✓ Codes shape: {tuple(codes.shape)}")
        print(f"    ✓ Time: {encode_time*1000:.2f}ms")
        
        # 2. Decode back to audio
        print("  Step 2: Decoding to audio...")
        start = time.time()
        with torch.no_grad():
            reconstructed = self.model.decode(codes)
        decode_time = time.time() - start
        print(f"    ✓ Reconstructed shape: {tuple(reconstructed.shape)}")
        print(f"    ✓ Time: {decode_time*1000:.2f}ms")
        
        # 3. Calculate quality metrics
        print("  Step 3: Quality metrics...")
        
        # Signal-to-Noise Ratio
        mse = F.mse_loss(reconstructed, waveform).item()
        snr = 20 * np.log10(1.0 / np.sqrt(mse))
        
        # Codebook usage
        codes_np = codes[0].cpu().numpy()
        usage = []
        for cb in range(codes.shape[-1]):
            unique = len(np.unique(codes_np[:, cb]))
            usage.append(unique / self.rvq_cfg.codebook_size * 100)
        
        print(f"\n📊 Results:")
        print(f"  Input:  {duration}s @ {self.audio_cfg.sample_rate}Hz = {samples} samples")
        print(f"  Codes:  {codes.shape[1]} frames × {codes.shape[2]} codebooks = {codes.numel()} tokens")
        print(f"  Compression ratio: {samples / codes.numel():.1f}:1")
        print(f"  SNR: {snr:.2f} dB")
        print(f"  MSE: {mse:.6f}")
        print(f"  Codebook usage: {np.mean(usage):.1f}%")
        
        return {
            'original': waveform,
            'reconstructed': reconstructed,
            'codes': codes,
            'snr': snr,
            'mse': mse,
            'usage': usage
        }
    
    # =========================================================================
    # EXPERIMENT 8: Codebook Health Monitor
    # =========================================================================
    
    def codebook_health(self, n_batches: int = 10):
        """Monitor codebook health during fake training."""
        print("\n" + "="*80)
        print("💊 EXPERIMENT 8: Codebook Health Monitoring")
        print("="*80)
        
        # Create monitor
        monitor = CodebookMonitor(self.rvq.n_codebooks, self.rvq_cfg.codebook_size)
        
        print(f"\n🩺 Monitoring {self.rvq.n_codebooks} codebooks over {n_batches} batches...")
        
        for batch in range(n_batches):
            # Generate random codes
            codes = torch.randint(
                0, self.rvq_cfg.codebook_size,
                (2, 50, self.rvq.n_codebooks)
            )
            
            # Update monitor
            monitor.update(codes)
            
            # Get report every few batches
            if batch % 3 == 0:
                report = monitor.report()
                print(f"\n  Batch {batch+1}:")
                for cb, (usage, perp) in enumerate(zip(report['usage_pct'], report['perplexity'])):
                    print(f"    CB{cb}: usage={usage:.1f}%, perplexity={perp:.1f}")
        
        final_report = monitor.report()
        
        print(f"\n📊 Final health report:")
        if final_report['collapse_warning']:
            print("  ⚠️ WARNING: Codebook collapse detected!")
        else:
            print("  ✅ Codebooks healthy")
        
        for cb, (usage, perp) in enumerate(zip(final_report['usage_pct'], final_report['perplexity'])):
            status = "✓" if usage > 20 else "⚠️"
            print(f"  {status} CB{cb}: usage={usage:.1f}%, perplexity={perp:.1f}")
        
        return final_report
    
    # =========================================================================
    # EXPERIMENT 9: Script Conditioning Test
    # =========================================================================
    
    def script_conditioning(self):
        """Test how different script families affect encoding."""
        print("\n" + "="*80)
        print("🔤 EXPERIMENT 9: Script Family Conditioning")
        print("="*80)
        
        # Create test audio (same for all scripts)
        samples = int(1.0 * self.audio_cfg.sample_rate)
        t = torch.linspace(0, 1.0, samples)
        test_audio = torch.sin(2 * np.pi * 440 * t).unsqueeze(0).unsqueeze(0).to(self.device)
        
        results = {}
        
        print("\n📝 Testing different script families:")
        for script_name, script_id in list(LANG_TO_SCRIPT.items())[:5]:  # Test first 5
            script_tensor = torch.tensor([script_id]).to(self.device)
            
            with torch.no_grad():
                # Encode with script conditioning
                codes = self.model.encode(test_audio, script_tensor)
                
                # Get adapter output
                adapter_out = self.script_adapter(script_tensor)
            
            results[script_name] = {
                'script_id': script_id,
                'codes': codes,
                'adapter_scale': adapter_out['scale'].mean().item(),
                'adapter_shift': adapter_out['shift'].mean().item()
            }
            
            print(f"\n  {script_name.upper()} (ID={script_id}):")
            print(f"    Scale: {results[script_name]['adapter_scale']:.3f}")
            print(f"    Shift: {results[script_name]['adapter_shift']:.3f}")
            print(f"    Codes shape: {tuple(codes.shape)}")
        
        # Compare code similarity
        print("\n🔄 Code similarity between scripts:")
        base_codes = results['hi']['codes']
        for script in ['bn', 'ta', 'en']:
            if script in results:
                sim = F.cosine_similarity(
                    base_codes.float().flatten(),
                    results[script]['codes'].float().flatten(),
                    dim=0
                ).item()
                print(f"  Hindi vs {script.upper()}: similarity={sim:.3f}")
        
        return results
    
    # =========================================================================
    # EXPERIMENT 10: Batch Processing
    # =========================================================================
    
    def batch_experiment(self, batch_sizes: list = [1, 2, 4, 8]):
        """Test batch processing performance."""
        print("\n" + "="*80)
        print("📦 EXPERIMENT 10: Batch Processing Performance")
        print("="*80)
        
        results = {}
        
        for batch_size in batch_sizes:
            if batch_size > 8:
                continue
                
            print(f"\n🔄 Testing batch size = {batch_size}")
            
            # Create batch
            samples = int(1.0 * self.audio_cfg.sample_rate)
            batch = torch.randn(batch_size, 1, samples).to(self.device)
            script_ids = torch.randint(0, 12, (batch_size,)).to(self.device)
            
            # Measure encode time
            torch.cuda.synchronize() if self.device.type == 'cuda' else None
            start = time.time()
            
            with torch.no_grad():
                codes = self.model.encode(batch, script_ids)
            
            torch.cuda.synchronize() if self.device.type == 'cuda' else None
            encode_time = time.time() - start
            
            # Measure decode time
            start = time.time()
            with torch.no_grad():
                reconstructed = self.model.decode(codes)
            torch.cuda.synchronize() if self.device.type == 'cuda' else None
            decode_time = time.time() - start
            
            results[batch_size] = {
                'encode_time': encode_time,
                'decode_time': decode_time,
                'total_time': encode_time + decode_time,
                'time_per_sample': (encode_time + decode_time) / batch_size,
                'codes_shape': tuple(codes.shape)
            }
            
            print(f"    Encode: {encode_time*1000:.1f}ms")
            print(f"    Decode: {decode_time*1000:.1f}ms")
            print(f"    Per sample: {results[batch_size]['time_per_sample']*1000:.1f}ms")
        
        # Plot scaling
        plt.figure(figsize=(10, 6))
        batch_sizes = list(results.keys())
        times = [results[bs]['total_time'] * 1000 for bs in batch_sizes]
        per_sample = [results[bs]['time_per_sample'] * 1000 for bs in batch_sizes]
        
        plt.plot(batch_sizes, times, 'o-', label='Total time', linewidth=2)
        plt.plot(batch_sizes, per_sample, 's-', label='Time per sample', linewidth=2)
        plt.xlabel("Batch Size")
        plt.ylabel("Time (ms)")
        plt.title("Batch Processing Performance")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(self.exp_dir / "batch_performance.png")
        plt.show()
        
        return results
    
    # =========================================================================
    # EXPERIMENT 11: Compare Presets
    # =========================================================================
    
    def compare_presets(self):
        """Compare different model presets."""
        print("\n" + "="*80)
        print("📏 EXPERIMENT 11: Model Preset Comparison")
        print("="*80)
        
        from lipika import _PRESETS, resolve_preset
        
        results = {}
        
        print("\n📊 Available presets:")
        for preset_name, preset in _PRESETS.items():
            print(f"\n  {preset_name.upper()}:")
            print(f"    {preset.label}")
            print(f"    Encoder channels: {preset.encoder_channels}")
            print(f"    Codebooks: {preset.n_codebooks} × {preset.codebook_size}")
            print(f"    Batch size: {preset.batch_size}")
            print(f"    Max duration: {preset.max_duration}s")
            
            # Create model with this preset
            audio_cfg = AudioConfig()
            rvq_cfg = RVQConfig(
                n_codebooks=preset.n_codebooks,
                codebook_size=preset.codebook_size
            )
            model_cfg = ModelConfig(
                encoder_channels=preset.encoder_channels,
                decoder_channels=preset.decoder_channels,
                disc_channels=preset.disc_channels,
                disc_depth=preset.disc_depth
            )
            
            model = LipikaTokenizer(audio_cfg, rvq_cfg, model_cfg, use_semantic_teacher=False)
            
            # Count parameters
            total_params = model.num_parameters()
            encoder_params = sum(p.numel() for p in model.encoder.parameters())
            rvq_params = sum(p.numel() for p in model.rvq.parameters())
            decoder_params = sum(p.numel() for p in model.decoder.parameters())
            
            results[preset_name] = {
                'total_params': total_params,
                'encoder_params': encoder_params,
                'rvq_params': rvq_params,
                'decoder_params': decoder_params,
                'frame_rate': model.frame_rate,
                'compression_ratio': model.encoder.compression_ratio
            }
            
            print(f"    Parameters: {total_params/1e6:.2f}M")
            print(f"      - Encoder: {encoder_params/1e6:.2f}M")
            print(f"      - RVQ: {rvq_params/1e6:.2f}M")
            print(f"      - Decoder: {decoder_params/1e6:.2f}M")
        
        return results
    
    # =========================================================================
    # EXPERIMENT 12: Performance Profile
    # =========================================================================
    
    def profile_performance(self, n_runs: int = 10):
        """Profile model performance."""
        print("\n" + "="*80)
        print("⏱️ EXPERIMENT 12: Performance Profiling")
        print("="*80)
        
        # Create test input
        samples = int(1.0 * self.audio_cfg.sample_rate)
        test_audio = torch.randn(1, 1, samples).to(self.device)
        script_id = torch.tensor([ScriptFamily.DEVANAGARI]).to(self.device)
        
        # Warmup
        for _ in range(3):
            _ = self.model.encode(test_audio, script_id)
        
        # Profile encode
        encode_times = []
        for _ in range(n_runs):
            torch.cuda.synchronize() if self.device.type == 'cuda' else None
            start = time.time()
            with torch.no_grad():
                codes = self.model.encode(test_audio, script_id)
            torch.cuda.synchronize() if self.device.type == 'cuda' else None
            encode_times.append(time.time() - start)
        
        # Profile decode
        decode_times = []
        for _ in range(n_runs):
            torch.cuda.synchronize() if self.device.type == 'cuda' else None
            start = time.time()
            with torch.no_grad():
                recon = self.model.decode(codes)
            torch.cuda.synchronize() if self.device.type == 'cuda' else None
            decode_times.append(time.time() - start)
        
        # Memory usage
        if self.device.type == 'cuda':
            memory_allocated = torch.cuda.memory_allocated() / 1e6
            memory_reserved = torch.cuda.memory_reserved() / 1e6
        else:
            memory_allocated = 0
            memory_reserved = 0
        
        print(f"\n📊 Performance metrics (over {n_runs} runs):")
        print(f"\n⏱️ Timing (1 second audio):")
        print(f"  Encode: {np.mean(encode_times)*1000:.2f} ± {np.std(encode_times)*1000:.2f} ms")
        print(f"  Decode: {np.mean(decode_times)*1000:.2f} ± {np.std(decode_times)*1000:.2f} ms")
        print(f"  Total:  {(np.mean(encode_times)+np.mean(decode_times))*1000:.2f} ms")
        print(f"  Real-time factor: {(np.mean(encode_times)+np.mean(decode_times))/1.0:.2f}x")
        
        print(f"\n💾 Memory:")
        print(f"  Model parameters: {self.model.num_parameters()/1e6:.2f}M")
        print(f"  Model size: {self.model.num_parameters()*4/1e6:.2f}MB (float32)")
        if self.device.type == 'cuda':
            print(f"  GPU allocated: {memory_allocated:.1f}MB")
            print(f"  GPU reserved: {memory_reserved:.1f}MB")
        
        print(f"\n📈 Throughput:")
        frames_per_sec = self.model.frame_rate
        print(f"  Frames/second: {frames_per_sec:.1f}")
        print(f"  Tokens/second: {frames_per_sec * self.rvq.n_codebooks:.1f}")
        print(f"  Audio/second: 1.0s (real-time)")
        
        return {
            'encode_times': encode_times,
            'decode_times': decode_times,
            'memory': memory_allocated,
            'frame_rate': frames_per_sec
        }


# =============================================================================
# MAIN INTERACTIVE LOOP
# =============================================================================

def main():
    """Interactive learning lab."""
    
    lab = LipikaLearningLab()
    
    while True:
        print("\n" + "="*80)
        print("🎓 LIPIKA LEARNING LAB - Interactive Mode")
        print("="*80)
        lab._print_menu()
        print("\n" + "-"*80)
        
        choice = input("\n🔬 Enter experiment number (1-12, or 'q' to quit): ").strip()
        
        if choice.lower() == 'q':
            print("\n👋 Goodbye! Happy learning!")
            break
        
        try:
            exp_num = int(choice)
            
            experiments = {
                1: lab.explore_model,
                2: lab.visualize_codebooks,
                3: lab.test_quantization,
                4: lab.watch_training,
                5: lab.analyze_discriminator,
                6: lab.loss_landscape,
                7: lab.encode_decode_demo,
                8: lab.codebook_health,
                9: lab.script_conditioning,
                10: lab.batch_experiment,
                11: lab.compare_presets,
                12: lab.profile_performance
            }
            
            if exp_num in experiments:
                print("\n" + "="*80)
                result = experiments[exp_num]()
                
                # Save result
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                result_file = lab.exp_dir / f"exp{exp_num}_{timestamp}.json"
                
                # Convert non-serializable items
                serializable = {}
                for k, v in result.items():
                    if isinstance(v, (np.ndarray, torch.Tensor)):
                        serializable[k] = v.tolist() if hasattr(v, 'tolist') else str(v)
                    elif isinstance(v, (np.integer, np.floating)):
                        serializable[k] = float(v)
                    else:
                        serializable[k] = v
                
                with open(result_file, 'w') as f:
                    json.dump(serializable, f, indent=2)
                
                print(f"\n💾 Results saved to: {result_file}")
                
            else:
                print(f"❌ Invalid experiment number: {exp_num}")
                
        except ValueError:
            print("❌ Please enter a number (1-12) or 'q'")
        except Exception as e:
            print(f"❌ Experiment failed: {e}")
            import traceback
            traceback.print_exc()
        
        input("\n⏎ Press Enter to continue...")


if __name__ == "__main__":
    main()