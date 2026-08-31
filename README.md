# AutoLyrics: Whisper Fine-Tuning for Lyrics Transcription 🎵🎙️

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)
![HuggingFace Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-orange.svg)
![PEFT](https://img.shields.io/badge/PEFT-LoRA-green.svg)

> **AutoLyrics** is an automated pipeline for fine-tuning OpenAI's `whisper-small` model specifically for Singing/Lyrics Transcription (SLT). Leveraging Low-Rank Adaptation (LoRA) via Hugging Face's PEFT library, the pipeline achieves highly efficient parameter-efficient fine-tuning (PEFT), smashing the project target of a $\ge$15% relative Word Error Rate (WER) reduction over the zero-shot baseline.

---

## 📑 Table of Contents
- [🔍 Overview](#-overview)
- [💾 Dataset Architecture](#-dataset-architecture)
- [⚙️ Preprocessing Pipeline](#-preprocessing-pipeline)
- [🧠 Model & Training Architecture](#-model--training-architecture)
- [📊 Results & Performance Evaluation](#-results--performance-evaluation)
- [📁 Project Directory Structure](#-project-directory-structure)
- [🚀 Quick Start](#-quick-start)

---

## 🔍 Overview

Singing transcription is notoriously difficult for standard speech ASR models due to continuous pitch tracking variations, prolonged vocal phonemes, and instrumental accompaniment blending. This repository bridges that gap using an optimization pipeline that:

1. Processes polyphonic audio through an advanced structural acoustic cleaning chain.
2. Injects LoRA adapters across all linear projections in both the Encoder and Decoder blocks of Whisper.
3. Automatically evaluates models against zero-shot baselines.
4. Generates comprehensive visual and structural performance reports (`results.json`, `performance_report.png`, and qualitative error logs).

---

## 💾 Dataset Architecture

The project utilizes the curated [gmenon/slt-lyrics-audio](https://huggingface.co/datasets/gmenon/slt-lyrics-audio) corpus on the Hugging Face Hub. 

* **Partition Strategy:** To guarantee strict scientific validation, datasets are split dynamically via unique **`singer` ID grouping**. This ensures complete acoustic isolation—no singer present in the training fold appears within the validation or test sets, eliminating identity leakage.
* **Split Configuration:**
  * **Train:** 70% of unique singers
  * **Validation:** 15% of unique singers
  * **Test:** 15% of unique singers

---

## ⚙️ Preprocessing Pipeline

To prepare highly dynamic polyphonic audio files for stable audio feature extraction, a multi-stage DSP and text-normalization pipeline is mapped out:

### 🔊 Audio Signal Normalization
* **DC Offset Suppression:** Centers the input waveform on the zero-amplitude axis to eliminate low-frequency signal drift.
* **Silence Aggressive Trimming:** Removes leading and trailing ambient silence frames utilizing an envelope power threshold of **30 dB** via `librosa`.
* **Integrated Loudness Normalization:** Standardizes multi-genre recording volumes to a uniform target of **`-23.0 LUFS`** matching the international **EBU R128** broadcast standard via `pyloudnorm`. Peak normalization down to `1.0 dB` headroom serves as an automatic safety fallback.

### 📝 Text Cleaning & Alignment
* **Structural Stripping:** Strips bracketed metadata out of the text transcript layout (e.g., matching `[Chorus]`, `(Verse 1)`, or background cues).
* **Formatting Simplification:** Lowercases the corpus and purges all punctuation arrays, explicitly shielding structural contractions (apostrophes) and hyphenated words.
* **Whitespace Compacting:** Flattens multiple space sequences into single space tokens.

---

## 🧠 Model & Training Architecture

* **Base Model backbone:** `openai/whisper-small`
* **PEFT Framework:** Low-Rank Adaptation (LoRA)
* **Target Adapter Placement:** `q_proj`, `k_proj`, `v_proj`, `out_proj`, `fc1`, `fc2` (Attention layers + Feed-Forward Network expansions)
* **Adapter Parameters:** Rank $r = 8$, Alpha $\alpha = 16$, LoRA Dropout = $0.1$
* **Optimization Setup:**
  * **Optimizer:** `AdamW` initialized at a base learning rate of **`2e-5`**
  * **Batch Tuning:** Physical `BATCH_SIZE = 4` reinforced with `GRAD_ACCUM_STEPS = 2` to yield an effective cross-entropy update window across 8 audio sequences simultaneously.
  * **Learning Rate Schedule:** Linear decay with a warmup cycle covering the first **10%** of total optimization iterations.
  * **Decoding Strategy:** Efficient Greedy Decoding layout (`num_beams = 1`).

---

## 📊 Results & Performance Evaluation

The model's fine-tuned weights were tested systematically against the frozen out-of-the-box base model on unseen target audio segments. Performance progress is monitored via standard Word Error Rate (WER) and Character Error Rate (CER) calculations.

### 📈 Quantitative Overview
The pipeline successfully converged, pulling validation cross-entropy loss down sharply and meeting the core performance goal:

| Evaluation Step | Cross-Entropy Loss | Word Error Rate (WER) | Character Error Rate (CER) |
| :--- | :---: | :---: | :---: |
| **Zero-Shot Baseline** | 6.7879 | 82.58% | 54.38% |
| **Best Validation Model** | 1.1484 | 67.94% | 45.03% |
| **Final Test Set** | **1.2259** | **62.94%** | **42.96%** |

* **Relative Baseline WER Reduction:** **`23.78%` relative improvement** (Successfully hitting the project target threshold of $>15\%$).

### 📊 Validation Curves
Visual tracking plots confirming early stabilization and parameter convergence are logged directly into your storage structure:

![Performance Report](report/performance_report.png)

### 🎙️ Qualitative Transcription Analysis
A snapshot look at real model output arrays reveals that even when the network encounters dense backing music, it tracks acoustic phonemes to make phonetic guesses instead of hallucinating formatting text loops:

```text
[Sample 03]
  REFERENCE : said why you always running in place
  PREDICTED : said why you're always running in place
  Metrics   : WER: 14.3%   CER: 8.3%   String Match: 96.0%

[Sample 04]
  REFERENCE : call out the troops now in a hurry
  PREDICTED : call out the truth now in a hurry
  Metrics   : WER: 12.5%   CER: 11.8%  String Match: 89.6%

[Sample 11]
  REFERENCE : for everything
  PREDICTED : for everything
  Metrics   : WER: 0.0%    CER: 0.0%   String Match: 100.0%

[Sample 17]
  REFERENCE : and getting nowhere
  PREDICTED : and getting nowhere
  Metrics   : WER: 0.0%    CER: 0.0%   String Match: 100.0%
