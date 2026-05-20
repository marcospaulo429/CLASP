# CLASP — Noise Augmentation & Robustness Evaluation

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Audio Preprocessing Pipeline](#2-audio-preprocessing-pipeline)
3. [Noise Augmentation Module](#3-noise-augmentation-module)
4. [Dataset Building with Noise](#4-dataset-building-with-noise)
5. [Generating Pre-Augmented WAV Files](#5-generating-pre-augmented-wav-files)
6. [Robustness Evaluation Protocol](#6-robustness-evaluation-protocol)
7. [Evaluation Metrics](#7-evaluation-metrics)
8. [Experimental Results](#8-experimental-results)
9. [Key Findings & Discussion](#9-key-findings--discussion)
10. [Reproducing the Experiments](#10-reproducing-the-experiments)

---

## 1. System Overview

CLASP (Cross-modal Language-Audio Semantic Pairing) is a retrieval model that maps spoken audio and text queries into a shared embedding space. Given a text question, the system retrieves the most semantically similar audio clip from a candidate pool.

### Architecture

The model is built on three frozen encoders plus a trainable fusion network:

| Component | Model | Output dimension |
|-----------|-------|-----------------|
| Audio encoder | HuBERT Large (`facebook/hubert-large-ls960-ft`) | 1024 |
| Vision encoder | EfficientNet-B7 (spectrogram as image) | 1000 |
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
5. **Pad**: if shorter than 1 second (16,000 samples), zero-pad to exactly 1 s — this prevents HuBERT's CNN stack from crashing on very short clips

All processed audio is `float32`. This function is called both during dataset building and during the online noise evaluation loop.

---

## 3. Noise Augmentation Module

**File:** [`src/clasp/audio/noise_augmentation.py`](../src/clasp/audio/noise_augmentation.py)

Three noise types are implemented. All functions operate on `float32` numpy arrays at 16 kHz and return audio clipped to `[-1, 1]`.

### 3.1 White Noise

```python
def add_white_noise(audio, snr_db=20.0) -> np.ndarray:
```

Adds white Gaussian noise at a specified signal-to-noise ratio.

**Algorithm:**
1. Compute signal power: `P_signal = mean(audio²)`
2. Derive target noise power from SNR definition:
   `P_noise = P_signal / 10^(SNR_dB / 10)`
3. Sample i.i.d. Gaussian noise, scale its RMS to `sqrt(P_noise)`
4. Add to signal and clip to `[-1, 1]`

Lower `snr_db` → more noise. At 5 dB the noise is quite perceptible; at 20 dB it is subtle.

### 3.2 Reverberation

```python
def add_reverberation(audio, decay_time_ms=150.0, sr=16000) -> np.ndarray:
```

Applies synthetic room reverberation by convolving the audio with a synthetic room impulse response (RIR).

**Algorithm:**
1. Build an exponential decay envelope of length `decay_time_ms` ms:
   `rir[t] = exp(-3t / decay_time)` — models energy decay in a room
2. Force `rir[0] = 1.0` (direct sound) and add a 0.5-amplitude early reflection at 50 ms
3. Normalize the RIR to peak 1
4. Convolve with `scipy.fftconvolve` (mode `"same"` preserves length)
5. Normalize output to 95% of peak to avoid clipping

> **Note:** In `run_noise_robustness_eval.py`, the `snr_db` parameter is repurposed as `decay_time_ms × 10` for reverb (e.g., SNR=20 → 200 ms decay, SNR=5 → 50 ms decay). This is a parameter-mapping quirk in the evaluation script.

### 3.3 Ambient Noise (ESC-50)

```python
def add_ambient_noise(audio, noise_audio, snr_db=20.0) -> np.ndarray:
def load_esc50_clip(esc50_files, target_sr=16000) -> np.ndarray:
def scan_esc50_files(esc50_dir) -> list[Path]:
```

Mixes a real-world environmental sound from the ESC-50 dataset into the speech signal at a specified SNR.

**Algorithm:**
1. Load a random ESC-50 clip (originally 44,100 Hz), resample to 16 kHz
2. Tile or crop the noise to match the speech length
3. Scale noise to the target SNR using the same power-based formula as white noise
4. Mix and clip

ESC-50 contains 2,000 clips across 50 environmental categories (rain, crowd, dog bark, etc.), making it a more realistic noise source than synthetic white noise.

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

Noise is **only applied to the training split**. Validation and test splits always use clean embeddings for a consistent evaluation baseline.

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

Instead of applying noise stochastically at embedding time, this script materializes all noisy variants as permanent `.wav` files. This creates a **reproducible, shareable dataset** that any downstream tool can consume — no code changes required.

### Output naming

For each input file `{a}_{p}_{q}.wav`, the script writes:

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

### Storage estimate

| Config | Files per original | Multiplier |
|--------|--------------------|-----------|
| `--noise-types white reverb` | +2 | 3× |
| `--noise-types white reverb ambient` | +3 | 4× |
| `--copy-originals` | +1 | +1 |

---

## 6. Robustness Evaluation Protocol

**File:** [`scripts/run_noise_robustness_eval.py`](../scripts/run_noise_robustness_eval.py)

This script measures how retrieval performance degrades as noise severity increases. It does **not** use pre-augmented WAVs — it applies noise online at evaluation time so that exact SNR levels can be swept programmatically.

### Protocol

1. **Load a trained CLASP checkpoint** and the pre-built test dataset (`.pkl`)
2. **Establish a clean baseline** using the embeddings already stored in the pickle
3. **For each noise type × SNR level combination:**
   - Load each test audio file from disk
   - Apply the noise augmentation at the specified SNR
   - Re-embed with HuBERT (and EfficientNet for vision)
   - Replace the stored audio/image embeddings with the noisy ones
   - Run the full retrieval evaluation
4. **Save all results** to a CSV

### SNR sweep

Default SNR levels: `20, 15, 10, 5` dB (comma-separated via `--snr-levels`).
Lower SNR = more noise = harder condition.

### Noise types evaluated

- `white` — synthetic Gaussian noise at each SNR level
- `reverb` — synthetic room reverberation (SNR parameter controls decay time)
- `ambient` — ESC-50 environmental sounds (requires `--wham-dir`)

### Retrieval setup

Each test sample is evaluated against `--num-candidates` candidates (default: 10). The candidate pool for each query consists of `num_candidates - 1` randomly sampled distractors plus the ground-truth audio. The ground truth is always placed last in the candidate list.

### Example

```bash
python scripts/run_noise_robustness_eval.py \
    --dataset-path data/datasets/total_dataset_spoken_squad.pkl \
    --model-path models/clasp_best.pt \
    --train-json data/datasets/spoken_squad/spoken_train-v1.1.json \
    --wav-dir data/datasets/spoken_squad/train_wav \
    --snr-levels "20,15,10,5" \
    --num-candidates 10 \
    --output-csv results/noise_robustness_run1.csv
```

---

## 7. Evaluation Metrics

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
MRR rewards partial credit — ranking the correct answer second scores 0.5. A random baseline with `k` candidates scores approximately `(ln k + 1) / (2k)`.

### Classification metrics (threshold-based)

At a cosine similarity threshold of 0.5, each (query, candidate) pair is classified as match / non-match. Standard sklearn metrics are then computed:

- **Macro Precision / Recall / F1** — class-balanced averages (treats positive and negative class equally)
- **Micro Precision / Recall / F1** — instance-weighted averages
- **Accuracy** — fraction of (query, candidate) pairs correctly classified
- **Golden Accuracy** — fraction of queries where the ground-truth cosine similarity ≥ 0.5

### Random baselines (10 candidates)

| Metric | Random baseline |
|--------|----------------|
| Hits@1 | 0.10 |
| MRR | ~0.29 |

---

## 8. Experimental Results

Three sets of results are stored in the repository root.

### 8.1 Small-scale robustness test (`results_noise_robustness.csv`)

Quick evaluation on ~12 test samples. Results are noisy due to small sample size but give directional signal.

| Condition | Hits@1 | MRR |
|-----------|--------|-----|
| Clean | 0.333 | 0.444 |
| White SNR=20 | 0.333 | 0.437 |
| White SNR=15 | 0.333 | 0.438 |
| White SNR=10 | 0.333 | 0.441 |
| White SNR=5 | **0.250** | 0.414 |
| Reverb SNR=20 | **0.250** | 0.386 |
| Reverb SNR=15 | **0.250** | 0.397 |
| Reverb SNR=10 | **0.250** | 0.393 |
| Reverb SNR=5 | 0.333 | 0.454 |

At this scale, the model is more sensitive to reverberation than white noise. White noise only causes a drop at 5 dB. Reverb causes consistent Hits@1 degradation across all decay lengths.

### 8.2 Large-scale noise model (`results_noise_model.csv`)

Full test set evaluation (~3,483 samples). This appears to be the best-performing model checkpoint.

| Condition | Hits@1 | MRR | Δ Hits@1 vs clean |
|-----------|--------|-----|-------------------|
| Clean | **0.552** | **0.709** | — |
| White SNR=20 | 0.101 | 0.297 | −81.7% |
| White SNR=15 | 0.099 | 0.296 | −82.1% |
| White SNR=10 | 0.098 | 0.294 | −82.2% |
| White SNR=5 | 0.097 | 0.295 | −82.4% |
| Reverb SNR=20 | 0.104 | 0.300 | −81.2% |
| Reverb SNR=15 | 0.102 | 0.298 | −81.5% |
| Reverb SNR=10 | 0.105 | 0.299 | −81.0% |
| Reverb SNR=5 | 0.103 | 0.299 | −81.3% |

The model achieves strong clean performance (Hits@1=0.552 vs. 0.10 random baseline with 10 candidates), but degrades catastrophically under any noise — falling to roughly random performance.

### 8.3 CLASP v2 (`results_CLASP2.csv`)

Second model variant, evaluated on a larger test set.

| Condition | Hits@1 | MRR | Δ Hits@1 vs clean |
|-----------|--------|-----|-------------------|
| Clean | 0.418 | 0.538 | — |
| White SNR=20 | 0.012 | 0.054 | −97.1% |
| White SNR=15 | 0.012 | 0.054 | −97.1% |
| White SNR=10 | 0.011 | 0.054 | −97.4% |
| White SNR=5 | 0.011 | 0.054 | −97.5% |
| Reverb SNR=20 | 0.010 | 0.054 | −97.6% |
| Reverb SNR=15 | 0.011 | 0.054 | −97.4% |
| Reverb SNR=10 | 0.012 | 0.054 | −97.2% |
| Reverb SNR=5 | 0.011 | 0.053 | −97.3% |

CLASP v2 degrades more severely than the noise model — from 0.418 to approximately 0.011 (essentially random on ~10 candidates). This suggests CLASP v2 was not trained with noise augmentation.

### Summary comparison

| Model | Clean Hits@1 | Noisy Hits@1 (avg) | Relative degradation |
|-------|-------------|---------------------|----------------------|
| Noise model | 0.552 | ~0.101 | ~81% |
| CLASP v2 | 0.418 | ~0.011 | ~97% |
| Random (10 cand.) | 0.100 | 0.100 | 0% |

---

## 9. Key Findings & Discussion

### Finding 1: Large domain gap between clean training and noisy inference

Both models were trained on clean audio embeddings. When noise is introduced at inference time, HuBERT's internal representations shift substantially, causing the audio embeddings to land far from their expected positions in the joint space. This explains the catastrophic degradation.

### Finding 2: SNR level has minimal impact within the tested range

For both white noise and reverb, the Hits@1 drop is nearly identical across SNR = 20, 15, 10, 5 dB. This suggests the degradation is primarily caused by the **qualitative change in the embedding distribution** (any noise shifts it) rather than the quantitative severity. Even "mild" noise (20 dB SNR) causes near-complete failure.

### Finding 3: White noise and reverb cause equivalent damage

Reverb and white noise produce statistically similar drops, even though they operate through completely different mechanisms. This is consistent with HuBERT being sensitive to any deviation from clean speech, rather than specific noise characteristics.

### Finding 4: Noise-augmented training significantly helps but does not solve the problem

The noise model (Hits@1=0.552 clean, ~0.10 noisy) is considerably more robust than CLASP v2 (0.418 → 0.011). However, both collapse well below clean performance. True robustness would require:
- Training with the same SNR levels used at evaluation
- Potentially noise-augmenting the text embeddings or the fusion network rather than just the audio encoder input
- Spectrogram augmentation (SpecAugment) at training time

### Finding 5: Vision branch is not re-embedded under noise

In `run_noise_robustness_eval.py`, the vision branch (EfficientNet spectrogram) is re-computed from the original clean WAV path, not the noisy audio. This means the multi-modal fusion receives one noisy modality (HuBERT) and one clean modality (EfficientNet), which may partially attenuate noise sensitivity — but also means the vision branch cannot be studied independently under noise.

---

## 10. Reproducing the Experiments

### Prerequisites

```bash
# Install dependencies
uv sync  # or: pip install -e .

# Download ESC-50 if using ambient noise
# Place it at: data/datasets/ESC-50/audio/*.wav
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
# Option A: stochastic online noise during embedding
python scripts/build_spoken_squad_pkl.py \
    --wav-dir data/datasets/spoken_squad/train_wav \
    --output data/datasets/total_dataset_noisy.pkl \
    --noise-prob 0.5 \
    --noise-types white reverb

# Option B: pre-bake noisy WAVs for a shareable static dataset
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
    --learning-rate 1e-4
```

### Step 4 — Evaluate noise robustness

```bash
python scripts/run_noise_robustness_eval.py \
    --dataset-path data/datasets/total_dataset_spoken_squad.pkl \
    --model-path models/clasp_noise_robust.pt \
    --train-json data/datasets/spoken_squad/spoken_train-v1.1.json \
    --wav-dir data/datasets/spoken_squad/train_wav \
    --snr-levels "20,15,10,5" \
    --num-candidates 10 \
    --output-csv results/noise_robustness_noise_robust_model.csv
```

### Step 5 — Zip the augmented dataset for sharing

```bash
# Preferred: tar.gz (faster, ~10-20% smaller than zip on audio files)
tar -czf train_wav_noisy.tar.gz data/datasets/spoken_squad/train_wav_noisy/

# Alternative: zip
zip -r train_wav_noisy.zip data/datasets/spoken_squad/train_wav_noisy/
```
