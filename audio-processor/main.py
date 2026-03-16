#!/usr/bin/env python3
"""
LIPIKA TOKENIZER - Complete Production Pipeline
================================================================================
A professional-grade audio processing pipeline that converts any audio file
into high-quality embeddings ready for downstream tasks.

Features:
- Universal audio format support (MP3, WAV, FLAC, OGG, OPUS, etc.)
- Automatic resampling to 24kHz
- Peak normalization to ±0.98
- Transformer-based encoding to 768-dim embeddings
- Batch processing
- Multiple model sizes (tiny, small, base, large)
- Export capabilities (ONNX, torch.save)
- Speaker verification
- Comprehensive logging
- Error handling
- Progress tracking
================================================================================
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Union
from dataclasses import dataclass, asdict
import json
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np

# Local imports
from audio import AudioPreprocessor
from encoder import AudioEncoder, create_audio_encoder, EncoderOutput

# ======================================================================
# CONFIGURATION & LOGGING
# ======================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the complete pipeline."""
    
    # Audio preprocessing
    target_sample_rate: int = 24000
    max_duration: float = 5.0
    peak_normalize: bool = True
    resample_quality: str = "soxr_hq"
    
    # Model configuration
    model_size: str = "base"  # tiny, small, base, large
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Output options
    save_embeddings: bool = True
    save_pooled: bool = True
    export_onnx: bool = False
    
    # Batch processing
    batch_size: int = 8
    
    # Feature flags
    enable_speaker_verification: bool = False
    enable_emotion_detection: bool = False
    enable_language_identification: bool = False
    
    def __post_init__(self):
        """Validate configuration."""
        if self.model_size not in ["tiny", "small", "base", "large"]:
            raise ValueError(f"model_size must be one of: tiny, small, base, large")
        
        if self.device not in ["cuda", "cpu"]:
            raise ValueError(f"device must be 'cuda' or 'cpu', got {self.device}")
        
        logger.info(f"Pipeline configured for {self.device.upper()}")
        logger.info(f"Model size: {self.model_size}")
        logger.info(f"Max duration: {self.max_duration}s at {self.target_sample_rate}Hz")


@dataclass
class ProcessingResult:
    """Results from processing a single audio file."""
    
    file_path: str
    success: bool
    duration: float
    waveform_shape: Tuple[int, ...]
    embeddings_shape: Tuple[int, ...]
    pooled_shape: Tuple[int, ...]
    processing_time: float
    error_message: Optional[str] = None
    speaker_embedding: Optional[torch.Tensor] = None
    metadata: Optional[Dict[str, Any]] = None
    
    # Optional tensors (not included in dict serialization)
    waveform: Optional[torch.Tensor] = None
    embeddings: Optional[torch.Tensor] = None
    pooled: Optional[torch.Tensor] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            'file_path': self.file_path,
            'success': self.success,
            'duration': self.duration,
            'waveform_shape': list(self.waveform_shape) if self.waveform_shape else [],
            'embeddings_shape': list(self.embeddings_shape) if self.embeddings_shape else [],
            'pooled_shape': list(self.pooled_shape) if self.pooled_shape else [],
            'processing_time': self.processing_time,
            'error_message': self.error_message,
            'metadata': self.metadata
        }
        
        # Convert speaker embedding to list if present
        if self.speaker_embedding is not None:
            result['speaker_embedding'] = self.speaker_embedding.cpu().tolist()
        
        return result


# ======================================================================
# PRODUCTION PIPELINE CLASS
# ======================================================================

class LipikaTokenizerPipeline:
    """
    Complete production pipeline for audio processing and encoding.
    
    This class integrates AudioPreprocessor and AudioEncoder into a single,
    production-ready pipeline with comprehensive features.
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the pipeline with configuration.
        
        Args:
            config: Pipeline configuration (uses defaults if None)
        """
        self.config = config or PipelineConfig()
        
        logger.info("=" * 70)
        logger.info("INITIALIZING LIPIKA TOKENIZER PIPELINE")
        logger.info("=" * 70)
        
        # Initialize components
        logger.info("Loading AudioPreprocessor...")
        self.preprocessor = AudioPreprocessor(
            target_sr=self.config.target_sample_rate,
            peak_norm=self.config.peak_normalize,
            resample_quality=self.config.resample_quality
        )
        
        logger.info(f"Loading AudioEncoder ({self.config.model_size})...")
        self.encoder = create_audio_encoder(
            model_size=self.config.model_size,
            sample_rate=self.config.target_sample_rate,
            window_seconds=self.config.max_duration
        )
        
        # Move encoder to device
        self.encoder = self.encoder.to(self.config.device)
        self.encoder.eval()  # Set to evaluation mode
        
        # Track statistics
        self.total_processed = 0
        self.total_time = 0.0
        self.results_history: List[ProcessingResult] = []
        
        logger.info(f"Pipeline ready on {self.config.device}")
        logger.info("=" * 70)
    
    def process_file(
        self,
        file_path: Union[str, Path],
        max_duration: Optional[float] = None,
        return_tensors: bool = True,
        extract_speaker_embedding: bool = False
    ) -> ProcessingResult:
        """
        Process a single audio file end-to-end.
        
        Args:
            file_path: Path to audio file
            max_duration: Override default max duration
            return_tensors: Return tensors in result
            extract_speaker_embedding: Extract speaker verification embedding
        
        Returns:
            ProcessingResult with all outputs and metadata
        """
        start_time = time.time()
        file_path = str(file_path)
        
        try:
            logger.info(f"Processing: {os.path.basename(file_path)}")
            
            # Step 1: Preprocess audio
            duration = max_duration or self.config.max_duration
            waveform = self.preprocessor.process(
                file_path,
                max_duration=duration
            )
            
            # Move to device
            waveform = waveform.to(self.config.device)
            
            # Step 2: Encode to embeddings
            with torch.no_grad():
                output = self.encoder(waveform, return_pooled=True)
            
            # Step 3: Extract speaker embedding if requested
            speaker_embedding = None
            if extract_speaker_embedding:
                # Use L2-normalized pooled output as speaker embedding
                speaker_embedding = F.normalize(output.pooled_output, p=2, dim=1)
            
            # Step 4: Gather metadata
            metadata = {
                'original_file': file_path,
                'file_size': os.path.getsize(file_path),
                'sample_rate': self.config.target_sample_rate,
                'model_size': self.config.model_size,
                'timestamp': datetime.now().isoformat()
            }
            
            # Add preprocessor metadata if available
            try:
                info = self.preprocessor.inspect(file_path)
                metadata.update({
                    'original_sr': info['original_sr'],
                    'original_channels': info['channels'],
                    'original_duration': info['duration_s'],
                    'original_format': info['format']
                })
            except Exception:
                pass
            
            # Step 5: Create result
            result = ProcessingResult(
                file_path=file_path,
                success=True,
                duration=duration,
                waveform_shape=tuple(waveform.shape),
                embeddings_shape=tuple(output.embeddings.shape),
                pooled_shape=tuple(output.pooled_output.shape),
                processing_time=time.time() - start_time,
                speaker_embedding=speaker_embedding,
                metadata=metadata
            )
            
            # Store tensors if requested
            if return_tensors:
                result.waveform = waveform.cpu()
                result.embeddings = output.embeddings.cpu()
                result.pooled = output.pooled_output.cpu()
            
            logger.info(f"Success: {output.embeddings.shape[1]} frames @ {output.embeddings.shape[2]}dim")
            logger.info(f"Time: {result.processing_time:.2f}s")
            
            # Update statistics
            self.total_processed += 1
            self.total_time += result.processing_time
            self.results_history.append(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed: {str(e)}")
            
            result = ProcessingResult(
                file_path=file_path,
                success=False,
                duration=0.0,
                waveform_shape=(),
                embeddings_shape=(),
                pooled_shape=(),
                processing_time=time.time() - start_time,
                error_message=str(e)
            )
            
            self.results_history.append(result)
            return result
    
    def process_batch(
        self,
        file_paths: List[Union[str, Path]],
        max_duration: Optional[float] = None,
        batch_size: Optional[int] = None
    ) -> List[ProcessingResult]:
        """
        Process multiple files in optimized batches.
        
        Args:
            file_paths: List of audio files
            max_duration: Override default max duration
            batch_size: Batch size (defaults to config.batch_size)
        
        Returns:
            List of ProcessingResult objects
        """
        batch_size = batch_size or self.config.batch_size
        results = []
        
        logger.info(f"Batch processing {len(file_paths)} files (batch size: {batch_size})")
        
        # Process in batches
        for i in range(0, len(file_paths), batch_size):
            batch_files = file_paths[i:i + batch_size]
            logger.info(f"Batch {i//batch_size + 1}/{(len(file_paths)-1)//batch_size + 1}")
            
            # Process each file in batch
            for file_path in batch_files:
                result = self.process_file(
                    file_path,
                    max_duration=max_duration,
                    return_tensors=True
                )
                results.append(result)
        
        # Calculate batch statistics
        success_count = sum(1 for r in results if r.success)
        logger.info(f"Batch complete: {success_count}/{len(file_paths)} successful")
        
        return results
    
    def verify_speakers(
        self,
        file1: Union[str, Path],
        file2: Union[str, Path],
        threshold: float = 0.7
    ) -> Tuple[bool, float]:
        """
        Verify if two audio files are from the same speaker.
        
        Args:
            file1: First audio file
            file2: Second audio file
            threshold: Similarity threshold (0-1)
        
        Returns:
            (is_same_speaker, similarity_score)
        """
        logger.info(f"Speaker verification: {os.path.basename(file1)} vs {os.path.basename(file2)}")
        
        # Process both files with speaker embedding extraction
        result1 = self.process_file(file1, extract_speaker_embedding=True)
        result2 = self.process_file(file2, extract_speaker_embedding=True)
        
        if not result1.success or not result2.success:
            logger.error("Failed to process one or both files")
            return False, 0.0
        
        # Calculate cosine similarity
        emb1 = result1.speaker_embedding.to(self.config.device)
        emb2 = result2.speaker_embedding.to(self.config.device)
        
        similarity = F.cosine_similarity(emb1, emb2).item()
        is_same = similarity > threshold
        
        logger.info(f"Similarity: {similarity:.4f} (threshold: {threshold})")
        logger.info(f"Result: {'SAME' if is_same else 'DIFFERENT'} speaker")
        
        return is_same, similarity
    
    def save_results(
        self,
        results: List[ProcessingResult],
        output_dir: Union[str, Path],
        save_tensors: bool = True,
        save_metadata: bool = True
    ):
        """
        Save processing results to disk.
        
        Args:
            results: List of ProcessingResult objects
            output_dir: Directory to save results
            save_tensors: Save tensor files (.pt)
            save_metadata: Save metadata JSON
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving results to {output_dir}")
        
        for i, result in enumerate(results):
            if not result.success:
                continue
            
            base_name = Path(result.file_path).stem
            
            # Save tensors
            if save_tensors and result.embeddings is not None:
                torch.save({
                    'embeddings': result.embeddings,
                    'pooled': result.pooled,
                    'waveform': result.waveform if result.waveform is not None else None,
                    'metadata': result.metadata
                }, output_dir / f"{base_name}_embeddings.pt")
            
            # Save speaker embedding if available
            if result.speaker_embedding is not None:
                torch.save(
                    result.speaker_embedding,
                    output_dir / f"{base_name}_speaker.pt"
                )
        
        # Save metadata JSON
        if save_metadata:
            metadata_list = [r.to_dict() for r in results]
            with open(output_dir / "processing_metadata.json", 'w', encoding='utf-8') as f:
                json.dump(metadata_list, f, indent=2)
        
        logger.info(f"Saved {len(results)} results")
    
    def export_model(self, output_path: Union[str, Path] = "audio_encoder.onnx"):
        """
        Export encoder to ONNX format for production deployment.
        
        Args:
            output_path: Path to save ONNX model
        """
        logger.info(f"Exporting model to ONNX: {output_path}")
        
        # Create dummy input
        dummy_input = torch.randn(1, 1, int(self.config.max_duration * self.config.target_sample_rate))
        dummy_input = dummy_input.to(self.config.device)
        
        # Export
        torch.onnx.export(
            self.encoder,
            dummy_input,
            output_path,
            input_names=['waveform'],
            output_names=['embeddings', 'pooled'],
            dynamic_axes={
                'waveform': {0: 'batch_size', 2: 'time'},
                'embeddings': {0: 'batch_size', 1: 'frames'},
                'pooled': {0: 'batch_size'}
            },
            opset_version=14
        )
        
        logger.info(f"Model exported to {output_path}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        if not self.results_history:
            return {}
        
        successful = [r for r in self.results_history if r.success]
        
        return {
            'total_processed': self.total_processed,
            'successful': len(successful),
            'failed': self.total_processed - len(successful),
            'average_time': self.total_time / max(1, self.total_processed),
            'total_time': self.total_time,
            'average_frames': float(np.mean([r.embeddings_shape[1] for r in successful if hasattr(r, 'embeddings_shape') and len(r.embeddings_shape) > 1])),
            'device': self.config.device,
            'model_size': self.config.model_size
        }
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        logger.info("Shutting down pipeline...")
        if hasattr(self, 'encoder'):
            self.encoder = self.encoder.cpu()
        logger.info(f"Processed {self.total_processed} files in {self.total_time:.2f}s")


# ======================================================================
# COMMAND-LINE INTERFACE
# ======================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    
    parser = argparse.ArgumentParser(
        description='Lipika Tokenizer - Production Audio Processing Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single file
  python main.py speech_hindi.wav
  
  # Process with custom duration and model
  python main.py speech_hindi.wav --duration 3.0 --model large
  
  # Batch process multiple files
  python main.py audio/*.wav --batch-size 16 --output-dir ./embeddings
  
  # Verify speakers
  python main.py --verify speaker1.wav speaker2.wav
  
  # Export model to ONNX
  python main.py --export-model
  
  # Process with all features
  python main.py speech_hindi.wav --save-embeddings --extract-speaker --metadata
        """
    )
    
    # Input arguments
    parser.add_argument(
        'input',
        nargs='*',
        help='Input audio file(s) or directory'
    )
    
    # Processing options
    parser.add_argument(
        '--duration', '-d',
        type=float,
        default=5.0,
        help='Maximum duration in seconds (default: 5.0)'
    )
    
    parser.add_argument(
        '--model', '-m',
        choices=['tiny', 'small', 'base', 'large'],
        default='base',
        help='Model size (default: base)'
    )
    
    parser.add_argument(
        '--device',
        choices=['cuda', 'cpu'],
        default=None,
        help='Device to use (auto-detected if not specified)'
    )
    
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=8,
        help='Batch size for processing (default: 8)'
    )
    
    # Output options
    parser.add_argument(
        '--output-dir', '-o',
        default='./output',
        help='Output directory (default: ./output)'
    )
    
    parser.add_argument(
        '--save-embeddings',
        action='store_true',
        help='Save embeddings to disk'
    )
    
    parser.add_argument(
        '--save-waveform',
        action='store_true',
        help='Save preprocessed waveform'
    )
    
    parser.add_argument(
        '--metadata',
        action='store_true',
        help='Save metadata JSON'
    )
    
    # Feature flags
    parser.add_argument(
        '--verify',
        nargs=2,
        metavar=('FILE1', 'FILE2'),
        help='Verify if two files are from the same speaker'
    )
    
    parser.add_argument(
        '--extract-speaker',
        action='store_true',
        help='Extract speaker verification embeddings'
    )
    
    parser.add_argument(
        '--export-model',
        action='store_true',
        help='Export encoder to ONNX format'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.7,
        help='Speaker verification threshold (default: 0.7)'
    )
    
    # Verbose output
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser


def main_cli():
    """Command-line interface entry point."""
    
    parser = create_parser()
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Determine device
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Handle special commands
    if args.export_model:
        config = PipelineConfig(
            model_size=args.model,
            max_duration=args.duration,
            device=device
        )
        with LipikaTokenizerPipeline(config) as pipeline:
            pipeline.export_model()
        return
    
    if args.verify:
        config = PipelineConfig(
            model_size=args.model,
            max_duration=args.duration,
            device=device
        )
        with LipikaTokenizerPipeline(config) as pipeline:
            is_same, similarity = pipeline.verify_speakers(
                args.verify[0],
                args.verify[1],
                threshold=args.threshold
            )
            print(f"\nSpeaker Verification Result:")
            print(f"  Similarity: {similarity:.4f}")
            print(f"  Same speaker: {is_same}")
        return
    
    # Process files
    if not args.input:
        parser.print_help()
        return
    
    # Collect input files
    input_files = []
    for pattern in args.input:
        path = Path(pattern)
        if path.is_dir():
            # Add all audio files from directory
            for ext in ['.wav', '.mp3', '.flac', '.ogg', '.opus', '.m4a', '.aiff', '.au']:
                input_files.extend(path.glob(f'*{ext}'))
        elif path.exists():
            input_files.append(path)
        else:
            logger.warning(f"File not found: {pattern}")
    
    if not input_files:
        logger.error("No valid input files found")
        return
    
    logger.info(f"Found {len(input_files)} files to process")
    
    # Create configuration
    config = PipelineConfig(
        target_sample_rate=24000,
        max_duration=args.duration,
        model_size=args.model,
        device=device,
        batch_size=args.batch_size,
        save_embeddings=args.save_embeddings,
        enable_speaker_verification=args.extract_speaker
    )
    
    # Process files
    with LipikaTokenizerPipeline(config) as pipeline:
        results = pipeline.process_batch(
            input_files,
            max_duration=args.duration,
            batch_size=args.batch_size
        )
        
        # Save results
        if args.save_embeddings or args.metadata:
            pipeline.save_results(
                results,
                args.output_dir,
                save_tensors=args.save_embeddings,
                save_metadata=args.metadata
            )
        
        # Print statistics
        stats = pipeline.get_statistics()
        print("\n" + "="*70)
        print("PIPELINE STATISTICS")
        print("="*70)
        for key, value in stats.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
        print("="*70)


# ======================================================================
# MAIN ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    main_cli()