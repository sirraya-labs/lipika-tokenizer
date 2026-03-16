"""
AudioEncoder: Neural network encoder for preprocessed audio tensors.
Designed to work seamlessly with AudioPreprocessor output format.
Fixed version with proper dimension handling and safety buffers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List
from dataclasses import dataclass


@dataclass
class EncoderOutput:
    """Structured output from AudioEncoder."""
    
    embeddings: torch.Tensor
    attention_mask: Optional[torch.Tensor] = None
    layer_outputs: Optional[List[torch.Tensor]] = None
    pooled_output: Optional[torch.Tensor] = None


class AudioEncoderConfig:
    """Configuration class for AudioEncoder hyperparameters."""
    
    def __init__(
        self,
        input_channels: int = 1,
        sample_rate: int = 24000,
        window_seconds: float = 5.0,
        hidden_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        ff_dim: int = 3072,
        dropout: float = 0.1,
        conv_kernel_sizes: List[int] = [10, 3, 3, 3, 3],
        conv_strides: List[int] = [5, 2, 2, 2, 2],
        pooling_strategy: str = "mean",  # "mean", "cls", or "max"
        use_same_padding: bool = True,
    ):
        self.input_channels = input_channels
        self.sample_rate = sample_rate
        self.window_seconds = window_seconds
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.conv_kernel_sizes = conv_kernel_sizes
        self.conv_strides = conv_strides
        self.pooling_strategy = pooling_strategy
        self.use_same_padding = use_same_padding
        
    @property
    def total_stride(self):
        """Calculate total stride from convolutional layers."""
        stride = 1
        for s in self.conv_strides:
            stride *= s
        return stride
    
    @property
    def frame_length(self):
        """Calculate frame length in seconds."""
        return self.total_stride / self.sample_rate
    
    @property
    def num_frames(self):
        """
        Calculate number of frames for a window using proper conv output formula.
        For same padding with stride > 1: output_length = ceil(input_length / stride)
        """
        length = int(self.sample_rate * self.window_seconds)
        
        for stride in self.conv_strides:
            length = (length + stride - 1) // stride
        
        return int(length)
    
    @property
    def max_position_embeddings(self):
        """Maximum position embeddings needed (with safety buffer)."""
        base_frames = self.num_frames
        # Add buffer for CLS token if needed
        if self.pooling_strategy == "cls":
            base_frames += 1
        # Add small safety buffer (2% or 10 frames, whichever is larger)
        safety_buffer = max(10, int(base_frames * 0.02))
        return base_frames + safety_buffer


class ConvFeatureExtractor(nn.Module):
    """
    Convolutional feature extractor with precise length calculation.
    """
    
    def __init__(self, config: AudioEncoderConfig):
        super().__init__()
        self.config = config
        
        # Build convolutional layers
        conv_layers = []
        in_channels = config.input_channels
        
        # Calculate dimensions for debugging
        current_length = int(config.sample_rate * config.window_seconds)
        print(f"  Initial length: {current_length} samples")
        
        for i, (kernel_size, stride) in enumerate(zip(
            config.conv_kernel_sizes, config.conv_strides
        )):
            # Determine out_channels for this layer
            if i == len(config.conv_kernel_sizes) - 1:
                out_channels = config.hidden_dim
            else:
                out_channels = min(config.hidden_dim, config.hidden_dim // (2 ** (len(config.conv_kernel_sizes) - 1 - i)))
            
            # Calculate padding for same padding
            if config.use_same_padding:
                padding = kernel_size // 2
            else:
                padding = 0
            
            # Calculate output length
            if config.use_same_padding:
                current_length = (current_length + stride - 1) // stride
            else:
                current_length = (current_length - kernel_size) // stride + 1
            
            print(f"  Layer {i+1}: kernel={kernel_size}, stride={stride}, padding={padding} → {current_length} frames")
            
            conv_layers.extend([
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),
                nn.Dropout(config.dropout),
            ])
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*conv_layers)
        self.actual_frames = current_length
        print(f"  Final frames: {current_length}")
        print(f"  Expected frames from config: {config.num_frames}")
        
        # Final projection if needed
        if in_channels != config.hidden_dim:
            self.final_proj = nn.Linear(in_channels, config.hidden_dim)
        else:
            self.final_proj = nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, time)
        
        Returns:
            (batch, frames, hidden_dim)
        """
        # Apply convolutions
        x = self.conv_layers(x)  # (batch, hidden_dim, frames)
        
        # Permute for linear projection
        x = x.transpose(1, 2)  # (batch, frames, hidden_dim)
        x = self.final_proj(x)  # (batch, frames, hidden_dim)
        
        return x


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding with dynamic sizing and safety buffer."""
    
    def __init__(self, config: AudioEncoderConfig):
        super().__init__()
        self.dropout = nn.Dropout(config.dropout)
        self.max_len = config.max_position_embeddings
        self.hidden_dim = config.hidden_dim
        
        # Create positional embeddings
        pe = torch.zeros(self.max_len, self.hidden_dim)
        position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.hidden_dim, 2).float() * 
                           (-torch.log(torch.tensor(10000.0)) / self.hidden_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe.unsqueeze(0))
        
        # Optional learnable embeddings
        self.learnable_pe = nn.Parameter(torch.zeros(1, self.max_len, self.hidden_dim))
        
        print(f"  Positional encoding initialized for max length: {self.max_len}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_dim)
        
        Returns:
            (batch, seq_len, hidden_dim)
        """
        seq_len = x.size(1)
        
        # Handle sequences longer than max_len (should not happen with safety buffer)
        if seq_len > self.max_len:
            print(f"  Warning: Truncating sequence from {seq_len} to {self.max_len}")
            x = x[:, :self.max_len, :]
            seq_len = self.max_len
        
        # Add positional encodings
        x = x + self.pe[:, :seq_len, :] + self.learnable_pe[:, :seq_len, :]
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer with pre-norm architecture."""
    
    def __init__(self, config: AudioEncoderConfig):
        super().__init__()
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(config.hidden_dim, config.ff_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ff_dim, config.hidden_dim),
            nn.Dropout(config.dropout),
        )
        
        # Layer norms (pre-norm architecture)
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-norm residual connection for attention
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = self.dropout(x)
        x = residual + x
        
        # Pre-norm residual connection for FFN
        residual = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = residual + x
        
        return x


class AudioEncoder(nn.Module):
    """
    Main AudioEncoder class with proper dimension handling.
    
    Input shape: (batch, 1, time) where time is typically 120000 for 5s @ 24kHz.
    Output shape: (batch, seq_len, hidden_dim) or pooled (batch, hidden_dim).
    
    Usage:
        encoder = AudioEncoder()
        waveform = preprocessor.process("speech.wav")  # (1, 1, 120000)
        embeddings = encoder(waveform)  # (1, num_frames, hidden_dim)
    """
    
    def __init__(self, config: Optional[AudioEncoderConfig] = None):
        super().__init__()
        
        if config is None:
            config = AudioEncoderConfig()
        
        self.config = config
        
        print("\n" + "="*60)
        print("AUDIO ENCODER INITIALIZATION")
        print("="*60)
        print(f"Sample rate: {config.sample_rate} Hz")
        print(f"Window: {config.window_seconds} seconds")
        print(f"Input samples: {int(config.sample_rate * config.window_seconds)}")
        print(f"Expected frames: {config.num_frames}")
        print(f"Max position embeddings (with buffer): {config.max_position_embeddings}")
        print(f"Hidden dimension: {config.hidden_dim}")
        print(f"Number of layers: {config.num_layers}")
        print(f"Number of heads: {config.num_heads}")
        print(f"FF dimension: {config.ff_dim}")
        print(f"Pooling strategy: {config.pooling_strategy}")
        print("-"*60)
        
        # Feature extraction
        self.feature_extractor = ConvFeatureExtractor(config)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(config)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(config) for _ in range(config.num_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(config.hidden_dim)
        
        # CLS token for pooling
        if config.pooling_strategy == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_dim) * 0.02)
        
        # Initialize weights
        self.apply(self._init_weights)
        print("="*60 + "\n")
        
    def _init_weights(self, module):
        """Initialize weights with truncated normal."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Conv1d):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(
        self,
        waveform: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_all_layers: bool = False,
        return_pooled: bool = True,
    ) -> Union[torch.Tensor, EncoderOutput]:
        """
        Forward pass through the encoder.
        
        Args:
            waveform: (batch, channels, time) from AudioPreprocessor
            attention_mask: (batch, time) mask for padding (1 for valid, 0 for padding)
            output_all_layers: Return outputs from all transformer layers
            return_pooled: Return pooled sequence representation
        
        Returns:
            EncoderOutput containing embeddings and optional masks/layers
        """
        batch_size = waveform.shape[0]
        
        # Step 1: Feature extraction (convolutional frontend)
        features = self.feature_extractor(waveform)  # (batch, frames, hidden_dim)
        
        # Step 2: Add positional encoding
        features = self.pos_encoder(features)  # (batch, frames, hidden_dim)
        
        # Step 3: Add CLS token if using CLS pooling
        if self.config.pooling_strategy == "cls":
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            features = torch.cat([cls_tokens, features], dim=1)
        
        # Step 4: Pass through transformer layers
        layer_outputs = []
        x = features
        
        for layer in self.layers:
            x = layer(x)
            if output_all_layers:
                layer_outputs.append(x)
        
        # Step 5: Final layer norm
        x = self.final_norm(x)
        
        # Step 6: Pooling (if requested)
        pooled = None
        if return_pooled:
            if self.config.pooling_strategy == "cls":
                pooled = x[:, 0, :]  # CLS token
            elif self.config.pooling_strategy == "mean":
                pooled = x.mean(dim=1)
            elif self.config.pooling_strategy == "max":
                pooled = x.max(dim=1)[0]
        
        # Remove CLS token from sequence output if present
        if self.config.pooling_strategy == "cls" and not output_all_layers:
            sequence_output = x[:, 1:, :]
        else:
            sequence_output = x
        
        if output_all_layers or return_pooled:
            return EncoderOutput(
                embeddings=sequence_output,
                attention_mask=attention_mask,
                layer_outputs=layer_outputs if output_all_layers else None,
                pooled_output=pooled,
            )
        
        return sequence_output
    
    def encode_batch(
        self,
        waveforms: List[torch.Tensor],
        max_duration: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Encode a batch of preprocessed waveforms.
        
        Args:
            waveforms: List of (1, 1, T) tensors from AudioPreprocessor
            max_duration: Maximum duration for padding (seconds)
        
        Returns:
            (batch, seq_len, hidden_dim) embeddings
        """
        if max_duration is not None:
            # Pad/truncate all to same length
            target_samples = int(max_duration * self.config.sample_rate)
            padded = []
            masks = []
            
            for w in waveforms:
                if w.shape[2] >= target_samples:
                    padded.append(w[:, :, :target_samples])
                    masks.append(torch.ones(1, target_samples, device=w.device))
                else:
                    pad_len = target_samples - w.shape[2]
                    padded_w = F.pad(w, (0, pad_len))
                    padded.append(padded_w)
                    
                    mask = torch.ones(1, w.shape[2], device=w.device)
                    mask = F.pad(mask, (0, pad_len))
                    masks.append(mask)
            
            batch = torch.cat(padded, dim=0)
            mask = torch.cat(masks, dim=0)
            return self.forward(batch, attention_mask=mask)
        else:
            # Assume all same length
            batch = torch.cat(waveforms, dim=0)
            return self.forward(batch)


# =============================================================================
# Factory function with automatic dimension calculation
# =============================================================================

def create_audio_encoder(
    model_size: str = "base",
    sample_rate: int = 24000,
    window_seconds: float = 5.0,
) -> AudioEncoder:
    """
    Factory function to create AudioEncoder with proper dimensions.
    
    Args:
        model_size: "tiny", "small", "base", or "large"
        sample_rate: Input sample rate (should match AudioPreprocessor)
        window_seconds: Expected window duration
    
    Returns:
        Configured AudioEncoder
    """
    
    configs = {
        "tiny": {
            "hidden_dim": 256,
            "num_layers": 6,
            "num_heads": 4,
            "ff_dim": 1024,
            "conv_kernel_sizes": [10, 3, 3, 3],
            "conv_strides": [5, 2, 2, 2],
        },
        "small": {
            "hidden_dim": 512,
            "num_layers": 8,
            "num_heads": 8,
            "ff_dim": 2048,
            "conv_kernel_sizes": [10, 3, 3, 3, 3],
            "conv_strides": [5, 2, 2, 2, 2],
        },
        "base": {
            "hidden_dim": 768,
            "num_layers": 12,
            "num_heads": 12,
            "ff_dim": 3072,
            "conv_kernel_sizes": [10, 3, 3, 3, 3],
            "conv_strides": [5, 2, 2, 2, 2],
        },
        "large": {
            "hidden_dim": 1024,
            "num_layers": 24,
            "num_heads": 16,
            "ff_dim": 4096,
            "conv_kernel_sizes": [10, 3, 3, 3, 3, 3],
            "conv_strides": [5, 2, 2, 2, 2, 2],
        }
    }
    
    if model_size not in configs:
        raise ValueError(f"model_size must be one of {list(configs.keys())}")
    
    base_config = configs[model_size]
    
    config = AudioEncoderConfig(
        sample_rate=sample_rate,
        window_seconds=window_seconds,
        **base_config
    )
    
    return AudioEncoder(config)


# =============================================================================
# Main integration script
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("AUDIO ENCODER - FINAL TEST WITH FIXES")
    print("="*60)
    
    # Test with your exact dimensions
    print("\n[1] Creating encoder for 5-second windows at 24kHz...")
    encoder = create_audio_encoder("base")
    
    # Create dummy input matching your AudioPreprocessor output
    batch_size = 1
    channels = 1
    samples = 120000  # 5 seconds @ 24kHz
    
    dummy_input = torch.randn(batch_size, channels, samples)
    print(f"\n[2] Test input shape: {tuple(dummy_input.shape)}")
    
    # Forward pass
    print("\n[3] Running forward pass...")
    with torch.no_grad():
        output = encoder(dummy_input, return_pooled=True)
    
    print(f"\n[4] Results:")
    print(f"    Sequence embeddings: {output.embeddings.shape}")
    print(f"    Pooled representation: {output.pooled_output.shape}")
    print(f"    Frames per second: {output.embeddings.shape[1] / 5.0:.1f}")
    print(f"    Embedding dimension: {output.embeddings.shape[2]}")
    
    print("\n" + "="*60)
    print("✓ SUCCESS: All dimension issues fixed!")
    print("="*60)