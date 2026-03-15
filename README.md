# Lipika Tokenizer

**A neural audio codec and tokenizer for Indic text-to-speech.**

Lipika converts raw speech waveforms into sequences of discrete integer tokens that can be fed directly into language models (VALL-E style), then reconstructs waveforms from those tokens. It is purpose-built for the phonetic demands of Indic languages — retroflex consonants, aspirated stops, tonal distinctions — and runs on CPU, Apple Silicon (MPS), and NVIDIA GPU without code changes.

---

## Table of Contents

- [Architecture](#architecture)
- [Supported Languages](#supported-languages)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Hardware Presets](#hardware-presets)
- [Training](#training)
  - [With your own data](#with-your-own-data)
  - [Resuming a run](#resuming-a-run)
  - [Multi-GPU training](#multi-gpu-training)
- [Inference](#inference)
- [Output Files](#output-files)
- [CLI Reference](#cli-reference)
- [Project Layout](#project-layout)
- [References](#references)

---

## Architecture

```
Waveform (24 kHz mono)
      |
  [ Stem Conv ]
      |
  [ Encoder Blocks x4 ]   <-- causal conv, dilation [1,3,9], strides [2,4,5,6]
      |                       compression ratio = 240x  =>  100 Hz frame rate
  [ Bottleneck ]
      |
  [ Script-Family Adapter ]  <-- AdaLN conditioning on Devanagari / Tamil / etc.
      |
  [ Residual VQ x N ]        <-- EMA codebook updates, dead-code reset
      |         |
  [ Semantic ]  [ Codes ]    <-- W2V-BERT distillation on codebook 0
  [ Head     ]
      |
  [ Decoder Blocks x4 ]      <-- causal ConvTranspose, strides [6,5,4,2]
      |
  Reconstructed Waveform
      |
  [ Discriminator ]          <-- Multi-Scale (MSD) + Multi-Period (MPD), hinge loss
```

**Key design choices:**

| Component | Design | Paper |
|---|---|---|
| Codec backbone | Causal conv encoder/decoder | EnCodec [1] |
| Quantizer | Residual VQ with EMA updates | SoundStream [2], VQ-VAE [3] |
| Semantic distillation | W2V-BERT-2.0 hidden states (layer 6) | data2vec [5], W2V-BERT [6] |
| Script conditioning | Adaptive Layer Norm on script-family embedding | AdaLN [12] |
| Discriminator | Multi-Scale + Multi-Period, hinge loss | MelGAN [7], HiFi-GAN [8] |
| Reconstruction loss | L1 waveform + mel-spectrogram + multi-resolution STFT | VITS [11], Vocos [13] |
| Codebook health | Dead-code reset when usage < threshold | [15] |

---

## Supported Languages

| Language | Code | Script |
|---|---|---|
| Hindi | `hi` | Devanagari |
| Marathi | `mr` | Devanagari |
| Sanskrit | `sa` | Devanagari |
| Nepali | `ne` | Devanagari |
| Konkani | `kok` | Devanagari |
| Bengali | `bn` | Bengali |
| Assamese | `as` | Bengali |
| Punjabi | `pa` | Gurmukhi |
| Gujarati | `gu` | Gujarati |
| Odia | `or` | Oriya |
| Tamil | `ta` | Tamil |
| Telugu | `te` | Telugu |
| Kannada | `kn` | Kannada |
| Malayalam | `ml` | Malayalam |
| Urdu | `ur` | Perso-Arabic |
| Kashmiri | `ks` | Perso-Arabic |
| Meitei | `mni` | Meitei |
| English (Indian) | `en` | Latin |

The script-family adapter gives the encoder a learned bias for each script group. Scripts with retroflex consonants (Devanagari, Tamil, Telugu, Kannada, Malayalam, Bengali, Gurmukhi, Oriya) receive an additional phonetic prior in their embedding initialisation.

---

## Requirements

**Python:** 3.9 or later (tested on 3.10, 3.12)

**Core (required):**
```
torch >= 2.0
numpy
```

**Recommended (enables full functionality):**
```
librosa          # audio resampling and mel filterbank
soundfile        # audio file I/O (.wav, .flac, .ogg, .mp3, .opus)
tqdm             # progress bars
tensorboard      # training curve logging
matplotlib       # training plot images
scipy            # signal processing utilities
transformers     # W2V-BERT semantic teacher (optional but improves quality)
```

**GPU training:**
- CUDA 12.1+ with at least 8 GB VRAM (`gpu-small` preset)
- 16 GB VRAM recommended for full EnCodec-scale training (`gpu-full` preset)

**CPU training:** fully supported, slower. The `cpu` preset reduces the model to ~12 M parameters for practical iteration speed.

**Apple Silicon:** MPS backend is auto-detected. Mixed precision is disabled on MPS (PyTorch limitation); use float32.

---

## Installation

```bash
# 1. Clone or copy tokenizer.py into your project directory

# 2. Install PyTorch (pick the right index URL for your CUDA version)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Install audio and training dependencies
pip install librosa soundfile tqdm tensorboard matplotlib scipy

# 4. (Optional) W2V-BERT semantic teacher — improves phoneme representation
pip install transformers

# 5. Verify everything works
python tokenizer.py smoke-test
```

The `smoke-test` command runs a full encode/decode forward pass for all three preset sizes and reports pass/fail with parameter counts. Run it after any installation change.

---

## Quick Start

### Smoke-test (no data needed)

```bash
python tokenizer.py smoke-test
```

Expected output:
```
[cpu-tiny]   PASSED  enc=64ch   cb=4x64  params=3.46M  codes=[1,100,4]  recon=[1,1,24000]
[gpu-small]  PASSED  enc=128ch  cb=4x64  params=11.3M  codes=[1,100,4]  recon=[1,1,24000]
[gpu-full]   PASSED  enc=256ch  cb=4x64  params=43.2M  codes=[1,100,4]  recon=[1,1,24000]
All smoke tests PASSED.
```

### Train on CPU (no GPU, no data)

If `./data` is empty, a synthetic dataset of sine-mix waveforms is used automatically — good for verifying the pipeline end-to-end.

```bash
python tokenizer.py train --preset cpu --epochs 10
```

### Train on GPU with your data

```bash
python tokenizer.py train \
    --data-dir ./data \
    --preset gpu-small \
    --epochs 200
```

### Encode an audio file to tokens

```bash
python tokenizer.py encode speech.wav \
    --checkpoint ./checkpoints/ckpt_step00010000.pt \
    --lang hi \
    --out tokens.pt
```

### Decode tokens back to audio

```bash
python tokenizer.py decode tokens.pt \
    --checkpoint ./checkpoints/ckpt_step00010000.pt \
    --out reconstructed.wav
```

---

## Hardware Presets

The `--preset` flag selects a model size and training configuration matched to your hardware. `auto` (default) picks automatically based on detected device and VRAM.

| Preset | Channels | Codebooks | Codebook size | Batch | VRAM | Approx. params |
|---|---|---|---|---|---|---|
| `cpu` | 64 | 4 | 256 | 2 | CPU only | ~12 M |
| `gpu-small` | 256 | 6 | 512 | 4 | ~8 GB | ~85 M |
| `gpu-full` | 512 | 8 | 1024 | 8 | ~16 GB | ~340 M |

**Auto-selection rules:**
- No CUDA GPU detected → `cpu`
- CUDA GPU with < 12 GB VRAM → `gpu-small`
- CUDA GPU with >= 12 GB VRAM → `gpu-full`

You can override any individual value after specifying a preset:

```bash
# Use cpu preset but with a larger batch
python tokenizer.py train --preset cpu --batch-size 4

# Use gpu-small but train twice as long
python tokenizer.py train --preset gpu-small --epochs 400

# Force a specific device
python tokenizer.py train --preset gpu-small --device cuda:1
```

---

## Training

### With your own data

Organise audio files under a single directory tree. Any combination of subdirectories is fine — the dataset crawls recursively.

```
data/
  hindi/
    speaker1/
      utt001.wav
      utt001.json     # optional metadata
      utt002.flac
  tamil/
    utt100.wav
    utt100.json
```

**Supported audio formats:** `.wav`, `.flac`, `.ogg`, `.mp3`, `.opus`

**Sidecar metadata (optional):** For each audio file, place a `.json` file with the same stem to specify the language. Without it, Devanagari (Hindi) is assumed.

```json
{ "lang": "ta" }
```

Run training:

```bash
python tokenizer.py train \
    --data-dir ./data \
    --checkpoint-dir ./checkpoints \
    --log-dir ./logs \
    --plot-dir ./plots \
    --output-dir ./outputs \
    --preset auto \
    --epochs 200 \
    --lr 3e-4
```

The first 2% of files (sorted deterministically by filename) are held out as a validation set. The split is reproducible — the same files are always validation regardless of how many training runs you start.

### Resuming a run

Lipika auto-resumes from the latest checkpoint in `--checkpoint-dir` on every run. To resume from a specific checkpoint:

```bash
python tokenizer.py train --resume ./checkpoints/ckpt_step00050000.pt --preset gpu-small
```

### Multi-GPU training

Multi-GPU DDP is supported on CUDA only. Pass `--gpus N`:

```bash
python tokenizer.py train --gpus 4 --preset gpu-full --data-dir ./data
```

DDP is launched via `torch.multiprocessing.spawn`. Only rank-0 writes checkpoints, logs, and plots.

### Disabling semantic distillation

W2V-BERT distillation requires the `transformers` package and adds ~315 M frozen parameters to memory. Disable it to save RAM or skip the `transformers` install:

```bash
python tokenizer.py train --preset cpu --no-semantic
```

---

## Inference

### Python API

```python
import torch
from tokenizer import LipikaTokenizer, AudioConfig, RVQConfig, ModelConfig

# Load a trained model
payload   = torch.load("checkpoints/ckpt_step00050000.pt", map_location="cpu", weights_only=False)
audio_cfg = AudioConfig(**payload["audio_cfg"])
rvq_cfg   = RVQConfig(**payload["rvq_cfg"])
model_cfg = ModelConfig(**payload["model_cfg"])

model = LipikaTokenizer(audio_cfg, rvq_cfg, model_cfg, use_semantic_teacher=False)
model.load_state_dict(payload["model_state"])
model.eval()

# Encode: waveform -> discrete tokens
# waveform shape: (batch, 1, samples) at 24 kHz, values in [-1, 1]
waveform  = torch.randn(1, 1, 24000)   # 1-second placeholder
script_id = torch.tensor([0])          # 0 = Devanagari

with torch.no_grad():
    codes = model.encode(waveform, script_id)
    # codes shape: (1, T_frames, n_codebooks)  e.g. (1, 100, 8) for 1s at 100Hz

# Decode: discrete tokens -> waveform
with torch.no_grad():
    reconstructed = model.decode(codes)
    # reconstructed shape: (1, 1, T_samples)

print(f"Tokens: {codes.shape}  |  Frame rate: {model.frame_rate} Hz")
print(f"Compression ratio: {model.encoder.compression_ratio}x")
```

### Language codes

Pass the ISO 639-1/3 code matching the language of the speech:

```python
from tokenizer import LANG_TO_SCRIPT, ScriptFamily

script_id = torch.tensor([int(LANG_TO_SCRIPT["ta"])])  # Tamil
```

---

## Output Files

During training, Lipika writes to four directories (all configurable):

```
checkpoints/
    ckpt_step00005000.pt    # model + optimiser state, rolling window of 5
    ckpt_step00010000.pt
    ...

logs/
    training.log            # full UTF-8 text log
    events.out.tfevents.*   # TensorBoard events

plots/
    training_curves_latest.png          # always up-to-date dashboard
    training_curves_step00000500.png    # snapshot every --plot-every steps
    spectrogram_latest.png              # real vs reconstructed mel comparison
    spectrogram_step00000500.png
    training_metrics.csv                # all scalars as a CSV

outputs/
    real_step00002000_sample0.wav           # ground-truth audio clips
    reconstructed_step00002000_sample0.wav  # model reconstructions
    ...
```

**Training dashboard** (`training_curves_latest.png`) shows:
- Generator total loss, reconstruction losses (L1 / mel / STFT)
- VQ commitment loss + W2V-BERT semantic distillation loss
- Discriminator loss, adversarial loss, feature-matching loss
- Learning rate curve
- Validation losses
- Per-codebook usage % and perplexity (collapse detection)

Open TensorBoard for interactive plots:

```bash
tensorboard --logdir ./logs
```

---

## CLI Reference

### `train`

```
python tokenizer.py train [options]

  --data-dir        PATH     Directory containing audio files (default: ./data)
  --checkpoint-dir  PATH     Where to save checkpoints (default: ./checkpoints)
  --log-dir         PATH     TensorBoard + text log directory (default: ./logs)
  --plot-dir        PATH     Training curve PNG output (default: ./plots)
  --output-dir      PATH     Audio sample output (default: ./outputs)

  --preset          PRESET   auto | cpu | gpu-small | gpu-full  (default: auto)
  --device          DEVICE   auto | cuda | cuda:N | mps | cpu   (default: auto)
  --epochs          INT      Number of training epochs (default: 200)
  --batch-size      INT      Override preset batch size
  --lr              FLOAT    Generator learning rate (default: 3e-4)
  --grad-accum      INT      Gradient accumulation steps (default: 1)
  --gpus            INT      Number of CUDA GPUs for DDP (default: 1)

  --no-semantic               Disable W2V-BERT semantic distillation
  --no-amp                    Disable automatic mixed precision
  --compile                   Enable torch.compile (CUDA only, PyTorch 2+)
  --resume          PATH      Resume from a specific checkpoint file

  --save-every      INT      Save checkpoint every N steps (default: 5000)
  --eval-every      INT      Run validation every N steps (default: 1000)
  --plot-every      INT      Save training plots every N steps (default: 500)
  --sample-every    INT      Save audio samples every N steps (default: 2000)
  --num-workers     INT      DataLoader worker processes (default: 0)
  --seed            INT      Random seed (default: 42)
```

### `encode`

```
python tokenizer.py encode AUDIO_PATH [options]

  AUDIO_PATH                  Input audio file (.wav, .flac, etc.)
  --checkpoint  PATH          Trained checkpoint (required)
  --lang        CODE          Language code, e.g. hi, ta, bn (default: hi)
  --out         PATH          Output .pt file for codes (default: codes.pt)
  --device      DEVICE        auto | cuda | mps | cpu (default: auto)
```

### `decode`

```
python tokenizer.py decode CODES_PATH [options]

  CODES_PATH                  Input .pt file containing codes tensor
  --checkpoint  PATH          Trained checkpoint (required)
  --out         PATH          Output .wav file (default: reconstructed.wav)
  --device      DEVICE        auto | cuda | mps | cpu (default: auto)
```

### `smoke-test`

```
python tokenizer.py smoke-test [options]

  --device      DEVICE        Device to run the test on (default: auto)
```

Runs a full forward pass for three model sizes and prints pass/fail. Always run this after installation and after any code change.

---

## Project Layout

```
tokenizer.py               # single-file implementation — everything is here
README.md                  # this file

data/                      # put your .wav / .flac files here
  speaker_id/
    utterance.wav
    utterance.json          # {"lang": "hi"}

checkpoints/               # auto-created during training
logs/                      # TensorBoard events + training.log
plots/                     # PNG training dashboards + CSV metrics
outputs/                   # real vs reconstructed .wav samples
```

---

## References

| # | Paper | Link |
|---|---|---|
| [1] | Défossez et al. (2022) — EnCodec: High Fidelity Neural Audio Compression | [arXiv:2210.13438](https://arxiv.org/abs/2210.13438) |
| [2] | Zeghidour et al. (2021) — SoundStream: End-to-End Neural Audio Codec | [arXiv:2107.03312](https://arxiv.org/abs/2107.03312) |
| [3] | van den Oord et al. (2017) — VQ-VAE: Neural Discrete Representation Learning | [arXiv:1711.00937](https://arxiv.org/abs/1711.00937) |
| [4] | Wang et al. (2023) — VALL-E: Neural Codec Language Models for Zero-Shot TTS | [arXiv:2301.02111](https://arxiv.org/abs/2301.02111) |
| [5] | Baevski et al. (2022) — data2vec: Self-Supervised Learning Framework | [arXiv:2202.03555](https://arxiv.org/abs/2202.03555) |
| [6] | Chung et al. (2021) — W2v-BERT: Contrastive + MLM for Speech Pre-Training | [arXiv:2108.06209](https://arxiv.org/abs/2108.06209) |
| [7] | Kumar et al. (2019) — MelGAN: GANs for Conditional Waveform Synthesis | [arXiv:1910.06711](https://arxiv.org/abs/1910.06711) |
| [8] | Kong et al. (2020) — HiFi-GAN: High Fidelity Speech Synthesis | [arXiv:2010.05646](https://arxiv.org/abs/2010.05646) |
| [9] | Gulrajani et al. (2017) — Improved Training of Wasserstein GANs | [arXiv:1704.00028](https://arxiv.org/abs/1704.00028) |
| [10] | Miyato et al. (2018) — Spectral Normalization for GANs | [arXiv:1802.05957](https://arxiv.org/abs/1802.05957) |
| [11] | Kim et al. (2021) — VITS: End-to-End TTS with Adversarial Learning | [arXiv:2106.06103](https://arxiv.org/abs/2106.06103) |
| [12] | Ba et al. (2016) — Layer Normalization | [arXiv:1607.06450](https://arxiv.org/abs/1607.06450) |
| [13] | Siuzdak (2023) — Vocos: Closing the Gap for Neural Vocoders | [arXiv:2306.00814](https://arxiv.org/abs/2306.00814) |
| [14] | Defossez et al. (2023) — AudioCraft / EnCodec with Language Model | [arXiv:2306.06189](https://arxiv.org/abs/2306.06189) |
| [15] | Zeyer et al. (2023) — Comprehensive Study of Codebook Collapse in VQ-VAEs | [arXiv:2309.12756](https://arxiv.org/abs/2309.12756) |