#!/usr/bin/env python3
"""
AudioPreprocessor Validation Test for speech_hindi.wav
===============================================================================
This script validates the AudioPreprocessor pipeline using a real Hindi speech
file, providing comprehensive diagnostic information about the preprocessing
stages and output tensor characteristics.
===============================================================================
"""

from audio import AudioPreprocessor
import os
import sys
import numpy as np

def main():
    # -------------------------------------------------------------------------
    # Stage 1: File Validation
    # -------------------------------------------------------------------------
    target_file = "speech_hindi.wav"
    
    print("=" * 80)
    print("AUDIO PREPROCESSOR VALIDATION REPORT")
    print("=" * 80)
    print(f"\n[STAGE 1] Input File Validation")
    print("-" * 40)
    
    if not os.path.exists(target_file):
        print(f"  STATUS: FAILED")
        print(f"  ERROR: Input file '{target_file}' not found")
        print(f"  WORKING DIRECTORY: {os.getcwd()}")
        print(f"\n  RECOMMENDED ACTION: Download a sample Hindi speech file:")
        print(f"  $ wget https://www.openslr.org/resources/121/hi_in_female.wav -O speech_hindi.wav")
        sys.exit(1)
    
    file_size = os.path.getsize(target_file) / 1024  # Size in KB
    print(f"  STATUS: PASSED")
    print(f"  FILE: {target_file}")
    print(f"  SIZE: {file_size:.2f} KB")
    print(f"  PATH: {os.path.abspath(target_file)}")

    # -------------------------------------------------------------------------
    # Stage 2: Preprocessor Initialization
    # -------------------------------------------------------------------------
    print(f"\n[STAGE 2] AudioPreprocessor Configuration")
    print("-" * 40)
    
    prep = AudioPreprocessor(
        target_sr=24000,
        peak_norm=True,
        resample_quality="soxr_hq"
    )
    
    print(f"  TARGET SAMPLE RATE: {prep.target_sr} Hz")
    print(f"  PEAK NORMALIZATION: Enabled (target: ±0.98)")
    print(f"  RESAMPLING QUALITY: soxr_hq (high-quality SoX resampler)")
    print(f"  SUPPORTED FORMATS: {', '.join(sorted(prep.SUPPORTED_EXTENSIONS))}")

    # -------------------------------------------------------------------------
    # Stage 3: File Inspection (Metadata without loading)
    # -------------------------------------------------------------------------
    print(f"\n[STAGE 3] Source File Metadata")
    print("-" * 40)
    
    try:
        metadata = prep.inspect(target_file)
        print(f"  ORIGINAL SAMPLE RATE: {metadata['original_sr']} Hz")
        print(f"  CHANNELS: {metadata['channels']} ({'mono' if metadata['channels'] == 1 else 'stereo'})")
        print(f"  DURATION: {metadata['duration_s']:.3f} seconds")
        print(f"  TOTAL SAMPLES: {metadata['original_samples']}")
        print(f"  FORMAT: {metadata['format']}")
        print(f"  ENCODING: {metadata['subtype']}")
        print(f"  RESAMPLING REQUIRED: {'Yes' if metadata['needs_resample'] else 'No'}")
        print(f"  OUTPUT SAMPLES (at 24kHz): {metadata['output_samples']}")
    except Exception as e:
        print(f"  ERROR during inspection: {str(e)}")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Stage 4: Audio Processing Pipeline
    # -------------------------------------------------------------------------
    print(f"\n[STAGE 4] Processing Pipeline Execution")
    print("-" * 40)
    print(f"  MAX DURATION: 5.0 seconds")
    print(f"  START OFFSET: 0.0 seconds")
    
    try:
        waveform = prep.process(target_file, max_duration=5.0)
        print(f"  STATUS: SUCCESS")
    except Exception as e:
        print(f"  STATUS: FAILED")
        print(f"  ERROR: {str(e)}")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Stage 5: Output Tensor Analysis
    # -------------------------------------------------------------------------
    print(f"\n[STAGE 5] Output Tensor Characteristics")
    print("-" * 40)
    
    # Basic tensor properties
    print(f"  TENSOR SHAPE: {tuple(waveform.shape)}")
    print(f"    - Batch dimension: {waveform.shape[0]} (single sample)")
    print(f"    - Channel dimension: {waveform.shape[1]} (mono audio)")
    print(f"    - Time dimension: {waveform.shape[2]} samples")
    
    actual_duration = waveform.shape[2] / prep.target_sr
    print(f"  ACTUAL DURATION: {actual_duration:.3f} seconds")
    print(f"  DATA TYPE: {waveform.dtype}")
    print(f"  DEVICE: {waveform.device}")
    
    # Statistical analysis
    print(f"\n  STATISTICAL ANALYSIS:")
    print(f"    - Mean: {waveform.mean().item():.8f}")
    print(f"    - Standard Deviation: {waveform.std().item():.8f}")
    print(f"    - Minimum Value: {waveform.min().item():.6f}")
    print(f"    - Maximum Value: {waveform.max().item():.6f}")
    print(f"    - Peak Absolute: {waveform.abs().max().item():.6f}")
    print(f"    - RMS: {torch.sqrt(torch.mean(waveform**2)).item():.8f}")
    
    # Normalization verification
    peak_value = waveform.abs().max().item()
    norm_error = abs(peak_value - 0.98) / 0.98 * 100
    print(f"\n  NORMALIZATION VERIFICATION:")
    print(f"    - Target peak: 0.980000")
    print(f"    - Actual peak: {peak_value:.6f}")
    print(f"    - Deviation: {norm_error:.4f}%")
    print(f"    - Status: {'✓ WITHIN TOLERANCE' if norm_error < 1.0 else '✗ OUT OF TOLERANCE'}")

    # -------------------------------------------------------------------------
    # Stage 6: Memory and Performance Metrics
    # -------------------------------------------------------------------------
    print(f"\n[STAGE 6] Resource Utilization")
    print("-" * 40)
    
    memory_bytes = waveform.element_size() * waveform.nelement()
    memory_kb = memory_bytes / 1024
    memory_mb = memory_kb / 1024
    
    print(f"  MEMORY FOOTPRINT:")
    print(f"    - Bytes: {memory_bytes} bytes")
    print(f"    - Kilobytes: {memory_kb:.2f} KB")
    print(f"    - Megabytes: {memory_mb:.4f} MB")
    
    sample_rate_ratio = metadata['original_sr'] / prep.target_sr
    compression_ratio = metadata['original_samples'] / waveform.shape[2]
    print(f"\n  PROCESSING METRICS:")
    print(f"    - Sample rate conversion: {metadata['original_sr']}Hz → {prep.target_sr}Hz")
    print(f"    - Sample rate ratio: {sample_rate_ratio:.4f}")
    print(f"    - Temporal compression ratio: {compression_ratio:.4f}")

    # -------------------------------------------------------------------------
    # Stage 7: Validation Summary
    # -------------------------------------------------------------------------
    print(f"\n[STAGE 7] Validation Summary")
    print("-" * 40)
    
    validation_checks = [
        ("File exists", True),
        ("Metadata extraction", True),
        ("Processing completed", True),
        ("Correct tensor shape", waveform.dim() == 3),
        ("Batch dimension = 1", waveform.shape[0] == 1),
        ("Channel dimension = 1", waveform.shape[1] == 1),
        ("Duration matches request", abs(actual_duration - 5.0) < 0.01),
        ("Peak normalization", abs(peak_value - 0.98) < 0.01),
        ("Float32 dtype", waveform.dtype == torch.float32),
    ]
    
    all_passed = True
    for check_name, check_result in validation_checks:
        status = "PASS" if check_result else "FAIL"
        if not check_result:
            all_passed = False
        print(f"  {status:4} : {check_name}")
    
    # -------------------------------------------------------------------------
    # Final Report
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("FINAL VERDICT:", end=" ")
    if all_passed:
        print("✓ AudioPreprocessor validation PASSED")
        print("\nThe preprocessed waveform meets all specifications and is ready")
        print("for input to the AudioEncoder module. The tensor has the correct")
        print("shape [1, 1, T], proper normalization (±0.98), and contains")
        print(f"{actual_duration:.3f} seconds of mono speech at {prep.target_sr} Hz.")
    else:
        print("✗ AudioPreprocessor validation FAILED")
        print("\nReview the failed checks above and verify the AudioPreprocessor")
        print("implementation or input file integrity.")
    print("=" * 80)

if __name__ == "__main__":
    import torch  # Import here to ensure availability
    main()