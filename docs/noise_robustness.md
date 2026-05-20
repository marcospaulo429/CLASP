# CLASP — Noise Augmentation & Robustness Evaluation

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Audio Preprocessing Pipeline](#2-audio-preprocessing-pipeline)
3. [Noise Augmentation Module](#3-noise-augmentation-module)
4. [Dataset Building with Noise](#4-dataset-building-with-noise)
5. [Generating Pre-Augmented WAV Files](#5-generating-pre-augmented-wav-files)
6. [Robustness Evaluation Protocol](#6-robustness-evaluation-protocol)
7. [Weights & Biases Integration](#7-weights--biases-integration)
8. [Evaluation Metrics](#8-evaluation-metrics)
9. [Experimental Results](#9-experimental-results)
10. [Key Findings & Discussion](#10-key-findings--discussion)
11. [Reproducing the Experiments](#11-reproducing-the-experiments)

---

## 1. System Overview

CLASP (Cross-modal Language-Audio Semantic Pairing) is a retrieval model that maps spoken audio and text queries into a shared embedding space. Given a text question, the system retrieves the most semantically similar audio clip from a candidate pool.

### Architecture

The model is built on three frozen encoders plus a trainable fusion network:

| Component | Model | Output dimension |
|-----------|-------|-----------------|
| Audio encoder | HuBERT Large (`facebook/hubert-large-ls960-ft`) | 1024 |
| Vision encoder | EfficientNet-B7 (log-spectrogram as image) | 1000 |
| Text encoder | LaBSE (`sentence-transformers/LaBSE`) | 768 |

The fusion network (`HubertLabseConcat`) takes the HuBERT embedding and the EfficientNet spectrogram embedding, projects each to 768 dimensions through independent MLP branches, concatenates them, then passes through a final MLP to produce a 768-dimensional joint audio representation:

```
HuBERT (1024)  →  [Linear→BN→ReLU→Dropout→Linear→BN→LeakyReLU→Dropout→Linear]  →  768
EfficientNet (1000)  →  [same structure]  →  768
         concatenate  →  [Linear→BN→LeakyReLU→Linear→LeakyReLU→Linear]  →  768
```

The final 768-dimensional output is compared against the LaBSE text embedding using cosine similarity.

### Dataset

**Spoken SQuAD** is used as the primary dataset. Each sample is a spoken question (WAV file) paired with its text transcription. WAV files are named positionally as `{article_idx}_{paragraph_idx}_{qa_idx}.wav`, matching their position in the JSON structure.

- Training split: 80%
- Validation split: 10%
- Test split: 10%

---

## 2. Audio Preprocessing Pipeline

**File:** [`src/clasp/inference/audio_preprocess.py`](../src/clasp/inference/audio_preprocess.py)

All audio — clean or noisy — passes through a single preprocessing function before embedding:

```python
def load_mono_16k_padded(file_path) -> np.ndarray:
```

Steps applied in order:

1. **Read**: load with `soundfile` (supports WAV, FLAC, OGG, etc.)
2. **Mono**: if multi-channel, average all channels
3. **Peak normalize**: divide by `max(|audio|)` so the waveform is in `[-1, 1]`
4. **Resample**: use `librosa` to bring to 16 kHz (HuBERT's expected sample rate)
5. **Pad**: if shorter than 1 second (16,000 samples), zero-pad to exactly 1 s — prevents HuBERT's CNN stack from crashing on very short clips

All processed audio is `float32`. This function is called both during dataset building and during the online noise evaluation loop.

---

## 3. Noise Augmentation Module

**File:** [`src/clasp/audio/noise_augmentation.py`](../src/clasp/audio/noise_augmentation.py)

The evaluation methodology follows **Tseng & Harwath (Interspeech 2025)** — "Probing the Robustness Properties of Neural Speech Codecs" — which sweeps three noise types across standardised level axes. All functions operate on `float32` numpy arrays at 16 kHz and return audio clipped to `[-1, 1]`.

### 3.1 White Noise

```python
def add_white_noise(audio, snr_db=20.0) -> np.ndarray:
```

Adds white Gaussian noise at a specified signal-to-noise ratio (SNR). This approximates sensor or transmission-induced broadband noise, matching the paper's white noise condition.

**Algorithm:**
1. Compute signal power: `P_signal = mean(audio²)`
2. Derive noise power: `P_noise = P_signal / 10^(SNR_dB / 10)`
3. Sample i.i.d. Gaussian noise and scale its RMS to `sqrt(P_noise)`
4. Add to signal and clip to `[-1, 1]`

**Axis:** SNR in dB. Paper sweeps −10 to +30 dB; our default is −10 to +30.

### 3.2 Reverberation

Two functions are available:

```python
def add_reverberation(audio, decay_time_ms=150.0, sr=16000) -> np.ndarray:
def add_reverberation_drr(audio, drr_db=0.0, decay_time_ms=150.0, sr=16000) -> np.ndarray:
```

`add_reverberation_drr` is the evaluation-facing function. It applies a synthetic room impulse response (RIR) parameterised by the **Direct-to-Reverberant Ratio (DRR)** — the same axis used by Tseng & Harwath.

**DRR definition:**
```
DRR = 10 × log10(P_direct / P_reverb)
```
Higher DRR → cleaner (more direct sound relative to reverb). The paper sweeps DRR from +10 down to −20 dB.

**Algorithm:**
1. Build an exponential decay reverb tail (no direct component): `tail[t] = exp(-3t / decay_time)`
2. Add a 0.5-amplitude early reflection at 50 ms to simulate wall bounces
3. Scale the tail so that `P_direct / P_tail = 10^(DRR/10)`, where `P_direct = 1` (unit impulse)
4. Combine: `RIR = direct_impulse + scaled_tail`
5. Convolve with `scipy.fftconvolve` (mode `"same"` preserves length) and normalise to 95% peak

`add_reverberation` (the legacy function) controls reverberation via decay time in milliseconds and is used during training-time augmentation and `augment_wavs.py`.

### 3.3 Ambient Noise

```python
def add_ambient_noise(audio, noise_audio, snr_db=20.0) -> np.ndarray:
def load_esc50_clip(esc50_files, target_sr=16000) -> np.ndarray:
def scan_esc50_files(esc50_dir) -> list[Path]:
```

Mixes a real-world environmental sound into the speech signal at a specified SNR. Supported sources:

- **ESC-50**: 2,000 clips across 50 categories (rain, crowd, dog bark, etc.). Files are automatically located in `{dir}/audio/*.wav` or `{dir}/*.wav`.
- **WHAM!**: Real-world background sounds used in the paper. Any directory of WAVs is accepted via `--ambient-dir` with `--ambient-source wham`.

**Algorithm:**
1. Load a random clip and resample to 16 kHz
2. Tile or crop to match the speech length (random crop start)
3. Scale to target SNR using the same power formula as white noise
4. Mix and clip

**Axis:** SNR in dB. Paper sweeps −10 to +30 dB; our default matches.

---

## 4. Dataset Building with Noise

**File:** [`scripts/build_spoken_squad_pkl.py`](../scripts/build_spoken_squad_pkl.py)

The dataset builder produces a `.pkl` file containing pre-computed embeddings for all three modalities. Noise augmentation can be applied **online** during embedding generation (training split only).

### Noise-aware embedding function

```python
def hubert_audio_files_with_noise(paths, processor, model, device,
                                   noise_prob, noise_snr, noise_types,
                                   esc50_files) -> list[torch.Tensor]:
```

For each audio file:
- With probability `noise_prob`, randomly pick a noise type from `noise_types`
- Apply the corresponding augmentation
- Embed with HuBERT: `mean(last_hidden_state, dim=1)` → shape `[1024]`

This stochastic approach means the training set sees both clean and noisy versions across epochs if `noise_prob < 1.0`, acting as data augmentation.

### Key CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--noise-prob` | `0.0` | Probability of augmenting each training sample |
| `--noise-snr` | `20.0` | SNR in dB for white/ambient noise |
| `--noise-types` | `white reverb` | Space-separated list of noise types to sample from |
| `--esc50-dir` | `None` | Required when `ambient` is in `--noise-types` |

Noise is **only applied to the training split**. Validation and test splits always use clean embeddings.

### Full example

```bash
python scripts/build_spoken_squad_pkl.py \
    --train-json data/datasets/spoken_squad/spoken_train-v1.1.json \
    --wav-dir data/datasets/spoken_squad/train_wav \
    --output data/datasets/total_dataset_noisy_train.pkl \
    --noise-prob 0.5 \
    --noise-snr 15 \
    --noise-types white reverb ambient \
    --esc50-dir data/datasets/ESC-50
```

---

## 5. Generating Pre-Augmented WAV Files

**File:** [`scripts/augment_wavs.py`](../scripts/augment_wavs.py)

Instead of applying noise stochastically at embedding time, this script materializes all noisy variants as permanent `.wav` files, creating a **reproducible, shareable dataset** that any downstream tool can consume.

### Output naming

For each input file `{a}_{p}_{q}.wav` the script writes:

```
{a}_{p}_{q}_white.wav
{a}_{p}_{q}_reverb.wav
{a}_{p}_{q}_ambient.wav
```

The originals are untouched. With `--copy-originals`, clean versions are also copied into the output directory, making it fully self-contained.

### Reproducibility

A fixed `--seed` controls `numpy.random` before the augmentation loop, ensuring the same ESC-50 clips are drawn and ambient noise is mixed identically across runs.

### Example workflows

**White + reverb only (no ESC-50 required):**
```bash
python scripts/augment_wavs.py \
    --wav-dir data/datasets/spoken_squad/train_wav \
    --out-dir data/datasets/spoken_squad/train_wav_noisy \
    --noise-types white reverb \
    --snr 20 \
    --copy-originals
```

**All three noise types:**
```bash
python scripts/augment_wavs.py \
    --wav-dir data/datasets/spoken_squad/train_wav \
    --out-dir data/datasets/spoken_squad/train_wav_noisy \
    --noise-types white reverb ambient \
    --snr 20 \
    --esc50-dir data/datasets/ESC-50 \
    --copy-originals \
    --seed 42
```

**Then build a unified dataset:**
```bash
python scripts/build_spoken_squad_pkl.py \
    --wav-dir data/datasets/spoken_squad/train_wav_noisy \
    --output data/datasets/total_dataset_noisy.pkl
```

**Zip for sharing:**
```bash
tar -czf train_wav_noisy.tar.gz data/datasets/spoken_squad/train_wav_noisy/
```

### Storage estimate

| Config | Files per original | Multiplier |
|--------|--------------------|-----------|
| `--noise-types white reverb` | +2 | 3× |
| `--noise-types white reverb ambient` | +3 | 4× |
| `--copy-originals` | +1 | +1 |

---

## 6. Robustness Evaluation Protocol

**File:** [`scripts/run_noise_robustness_eval.py`](../scripts/run_noise_robustness_eval.py)

This script measures how retrieval performance degrades as noise severity increases, following the methodology of Tseng & Harwath (Interspeech 2025). It applies noise online at evaluation time so that exact level grids can be swept without storing pre-augmented audio.

### Key design decisions aligned with the paper

| Aspect | Paper | This implementation |
|--------|-------|---------------------|
| White noise axis | SNR (dB) | `--snr-levels` (default: `30,20,10,5,0,-5,-10`) |
| Ambient noise axis | SNR (dB) | `--snr-levels` (same parameter) |
| Reverb axis | DRR (dB) | `--drr-levels` (default: `10,5,0,-5,-10,-15,-20`) |
| Reverb implementation | DNS Challenge real RIRs | Synthetic exponential-decay RIR scaled to target DRR |
| Ambient source | WHAM! | ESC-50 or WHAM! (selectable via `--ambient-source`) |
| Vision branch under noise | N/A (codec study) | Both HuBERT **and** EfficientNet receive the same noisy waveform |

> **Reverb vs SNR**: earlier versions of this codebase incorrectly reused the `--snr-levels` parameter for reverberation, mapping SNR → `decay_time_ms × 10`. This has been corrected: reverberation now uses a separate `--drr-levels` parameter and the `add_reverberation_drr` function, which properly scales the RIR tail to hit the target DRR.

### Protocol

1. **Load** a trained CLASP checkpoint and the pre-built test dataset (`.pkl`)
2. **Establish a clean baseline** using the embeddings already stored in the pickle (no re-embedding needed)
3. **White and ambient noise** — for each SNR level:
   - Load each test audio file from disk
   - Apply noise at the specified SNR
   - Re-embed with **both** HuBERT and EfficientNet (spectrogram of the noisy audio)
   - Run the full retrieval evaluation
4. **Reverberation** — same steps, but sweep DRR levels instead of SNR
5. **Log and save** results to CSV and optionally to W&B

### Retrieval setup

Each test sample is evaluated against `--num-candidates` candidates (default: 10). The candidate pool consists of `num_candidates - 1` randomly sampled distractors plus the ground-truth audio (placed last). The ground truth's rank is measured and averaged across all queries.

### Full example

```bash
python scripts/run_noise_robustness_eval.py \
    --dataset-path data/datasets/total_dataset_spoken_squad.pkl \
    --model-path models/clasp_best.pt \
    --train-json data/datasets/spoken_squad/spoken_train-v1.1.json \
    --wav-dir data/datasets/spoken_squad/train_wav \
    --snr-levels "30,20,10,5,0,-5,-10" \
    --drr-levels "10,5,0,-5,-10,-15,-20" \
    --ambient-dir data/datasets/ESC-50 \
    --ambient-source esc50 \
    --num-candidates 10 \
    --output-csv results/robustness.csv \
    --wandb-project clasp-paper \
    --wandb-run-name "4090-robustness-run1"
```

---

## 7. Weights & Biases Integration

Both the training script and the robustness evaluation script support optional W&B logging via three shared flags:

| Flag | Description |
|------|-------------|
| `--wandb-project` | W&B project name. Omit to disable all W&B logging. |
| `--wandb-entity` | W&B entity (username or team). |
| `--wandb-run-name` | Display name for the run in the W&B dashboard. |

### What gets logged during robustness evaluation

**Per condition** (one `wandb.log` call per noise type × level):
- `white/Hits@1`, `white/MRR`, `white/Macro_F1`, `white/Accuracy`, `white/Golden_Accuracy`
- Same keys under `ambient/` and `reverb/`
- `snr_db` (for white/ambient conditions) or `drr_db` (for reverb) — used as the x-axis in W&B line charts

**Clean baseline** (logged once, prefixed with `clean/`):
- `clean/Hits@1`, `clean/MRR`, etc.

**Full results table** (`wandb.Table`):
- Columns: `noise_config`, `noise_type`, `level_axis`, `level`, plus all metric columns
- One row per condition; can be used to build custom charts in the W&B UI

**Run summary** (visible in the runs list without opening the run):
- `clean_Hits@1`, `clean_MRR`
- `min_noisy_Hits@1`, `max_noisy_Hits@1`
- `avg_relative_degradation` — `1 - mean(noisy_Hits@1) / clean_Hits@1`

### What gets logged during training

Per epoch:
- `train_loss`, `val_loss`, `epoch`

Run config: all CLI arguments (learning rate, batch size, temperature, mode, etc.).

---

## 8. Evaluation Metrics

**File:** [`src/clasp/evaluation/metrics.py`](../src/clasp/evaluation/metrics.py)

All metrics are computed by `evaluate_model_on_candidates`.

### Retrieval metrics

**Hits@1** — fraction of queries where the ground-truth audio is ranked first:
```
Hits@1 = (# queries where rank(ground truth) = 1) / (# total queries)
```
This is the primary metric for retrieval quality. A random baseline with `k` candidates scores `1/k`.

**MRR (Mean Reciprocal Rank)** — average of `1/rank` across all queries:
```
MRR = (1/N) × Σ 1/rank(ground_truth_i)
```
MRR rewards partial credit — ranking the correct answer second scores 0.5.

### Classification metrics (threshold-based)

At a cosine similarity threshold of 0.5, each (query, candidate) pair is classified as match / non-match:

- **Macro Precision / Recall / F1** — class-balanced averages
- **Micro Precision / Recall / F1** — instance-weighted averages
- **Accuracy** — fraction of pairs correctly classified
- **Golden Accuracy** — fraction of queries where the ground-truth cosine similarity ≥ 0.5

### Random baselines (10 candidates)

| Metric | Random baseline |
|--------|----------------|
| Hits@1 | 0.10 |
| MRR | ~0.29 |

---

## 9. Experimental Results

Three CSV files in the repository root capture results from earlier evaluation runs. Note that these were generated with the **old evaluation script** (reverb parameterised by SNR, not DRR; vision branch using clean audio). They are preserved for reference.

### 9.1 Small-scale test (`results_noise_robustness.csv`)

~12 test samples; results are directional only.

| Condition | Hits@1 | MRR |
|-----------|--------|-----|
| Clean | 0.333 | 0.444 |
| White SNR=20 | 0.333 | 0.437 |
| White SNR=5 | **0.250** | 0.414 |
| Reverb (old, SNR=20) | **0.250** | 0.386 |
| Reverb (old, SNR=5) | 0.333 | 0.454 |

### 9.2 Noise-augmented model (`results_noise_model.csv`)

~3,483 test samples. Best-performing checkpoint.

| Condition | Hits@1 | MRR | Δ Hits@1 |
|-----------|--------|-----|----------|
| Clean | **0.552** | **0.709** | — |
| White SNR=20 | 0.101 | 0.297 | −81.7% |
| White SNR=5 | 0.097 | 0.295 | −82.4% |
| Reverb (old, SNR=20) | 0.104 | 0.300 | −81.2% |
| Reverb (old, SNR=5) | 0.103 | 0.299 | −81.3% |

### 9.3 CLASP v2 (`results_CLASP2.csv`)

Larger test set; model not trained with noise augmentation.

| Condition | Hits@1 | MRR | Δ Hits@1 |
|-----------|--------|-----|----------|
| Clean | 0.418 | 0.538 | — |
| White SNR=20 | 0.012 | 0.054 | −97.1% |
| Reverb (old, SNR=20) | 0.010 | 0.054 | −97.6% |

### Summary comparison

| Model | Clean Hits@1 | Noisy Hits@1 (avg) | Degradation |
|-------|-------------|---------------------|-------------|
| Noise model | 0.552 | ~0.101 | ~81% |
| CLASP v2 | 0.418 | ~0.011 | ~97% |
| Random (10 cand.) | 0.100 | 0.100 | 0% |

---

## 10. Key Findings & Discussion

### Finding 1: Any noise causes catastrophic degradation

Both models collapse to near-random performance (Hits@1 ≈ 0.10) under any noise, even at mild SNR (20 dB). This indicates a large domain gap between the clean-speech distribution that HuBERT was fine-tuned on and the noisy-speech distribution. The embedding space shifts enough that the fusion network can no longer find the correct match.

### Finding 2: Noise level has minimal marginal impact

Within the tested SNR range (5–20 dB), Hits@1 barely changes. The step-change happens between clean and any noise, not between mild and severe noise. This suggests the model's failure mode is qualitative (wrong embedding region), not quantitative (wrong distance).

### Finding 3: White noise and reverb cause equivalent damage

Despite operating through completely different mechanisms, both noise types produce statistically similar Hits@1 drops. This is consistent with HuBERT's sensitivity to any deviation from clean speech rather than specific noise characteristics.

### Finding 4: Noise-augmented training helps but doesn't close the gap

The noise model (Hits@1=0.552 clean → ~0.10 noisy, −81%) is significantly more robust than CLASP v2 (0.418 → 0.011, −97%). However, both collapse well below clean performance. True robustness would likely require:
- Training at the same SNR/DRR levels used at evaluation
- Augmenting the spectrogram branch independently (SpecAugment)
- Noise-conditioning the fusion head, not just the audio encoder input

### Finding 5 (corrected): Both encoder branches now use noisy audio

Earlier results had a methodological inconsistency: the EfficientNet spectrogram branch used the original clean audio path even during noisy evaluation, meaning the fusion network received one noisy modality and one clean modality. This has been fixed — `efficientnet_embeddings_from_audio_arrays` now accepts in-memory numpy arrays so both HuBERT and EfficientNet receive the same noisy waveform. **Results from future runs will not be directly comparable to the three archived CSVs above.**

### Relation to Tseng & Harwath (Interspeech 2025)

The paper evaluates neural speech **codecs** (encoder → RVQ → decoder) using PESQ, ASR-WER, ASV-EER, and SER-ACC. CLASP is a cross-modal **retrieval** system using Hits@1 and MRR. The evaluation conditions (noise types, level axes, level grids) are now aligned with the paper's protocol. The key qualitative finding mirrors the paper: training data diversity and model design choices matter far more than operating at a higher bitrate (or, in CLASP's case, a higher capacity fusion head) for achieving noise robustness.

---

## 11. Reproducing the Experiments

### Prerequisites

```bash
# Install all dependencies (includes wandb)
uv sync  # or: pip install -e .

# Download ESC-50 if using ambient noise
# Place WAVs at: data/datasets/ESC-50/audio/*.wav
```

### Step 1 — Build clean dataset

```bash
python scripts/build_spoken_squad_pkl.py \
    --train-json data/datasets/spoken_squad/spoken_train-v1.1.json \
    --wav-dir data/datasets/spoken_squad/train_wav \
    --output data/datasets/total_dataset_spoken_squad.pkl
```

### Step 2 — Build noise-augmented dataset (optional)

```bash
# Option A: stochastic online noise during embedding (fast, variable)
python scripts/build_spoken_squad_pkl.py \
    --wav-dir data/datasets/spoken_squad/train_wav \
    --output data/datasets/total_dataset_noisy.pkl \
    --noise-prob 0.5 \
    --noise-types white reverb

# Option B: pre-bake noisy WAVs (reproducible, shareable)
python scripts/augment_wavs.py \
    --wav-dir data/datasets/spoken_squad/train_wav \
    --out-dir data/datasets/spoken_squad/train_wav_noisy \
    --noise-types white reverb ambient \
    --snr 20 \
    --esc50-dir data/datasets/ESC-50 \
    --copy-originals \
    --seed 42

python scripts/build_spoken_squad_pkl.py \
    --wav-dir data/datasets/spoken_squad/train_wav_noisy \
    --output data/datasets/total_dataset_noisy.pkl
```

### Step 3 — Train

```bash
python scripts/train.py \
    --dataset-path data/datasets/total_dataset_noisy.pkl \
    --save-path models/clasp_noise_robust.pt \
    --num-epochs 100 \
    --learning-rate 1e-4 \
    --wandb-project clasp-paper \
    --wandb-run-name "noise-robust-v1"
```

### Step 4 — Evaluate noise robustness

```bash
python scripts/run_noise_robustness_eval.py \
    --dataset-path data/datasets/total_dataset_spoken_squad.pkl \
    --model-path models/clasp_noise_robust.pt \
    --train-json data/datasets/spoken_squad/spoken_train-v1.1.json \
    --wav-dir data/datasets/spoken_squad/train_wav \
    --snr-levels "30,20,10,5,0,-5,-10" \
    --drr-levels "10,5,0,-5,-10,-15,-20" \
    --ambient-dir data/datasets/ESC-50 \
    --ambient-source esc50 \
    --num-candidates 10 \
    --output-csv results/robustness.csv \
    --wandb-project clasp-paper \
    --wandb-run-name "4090-robustness-run1"
```

### Step 5 — Share the augmented dataset

```bash
# tar.gz is faster and ~10-20% smaller than zip on audio files
tar -czf train_wav_noisy.tar.gz data/datasets/spoken_squad/train_wav_noisy/

# Unpack on another machine
tar -xzf train_wav_noisy.tar.gz
```
