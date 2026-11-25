# Evaluating Spatial Reasoning and Visual Robustness in Small Vision-Language Models

A comparative study of six small vision-language models (VLMs) on controlled yes/no spatial questions.
This repository holds the code, data, raw predictions and metrics behind the report.

**Authors:** Nadia Binte Asif and Maneesha Vanga, University of Tennessee at Chattanooga  
**Course:** 7910R, Foundational Models

## Overview

Image accuracy alone does not show whether a model reads spatial layout or repeats one answer.
Each scene in this benchmark is therefore paired with controlled edits:

- **Spatial change:** the object arrangement changes, so the correct answer flips.
- **Label-preserving change:** brightness, clutter or occlusion, so the correct answer stays the same.

A model should change its answer for the first kind and keep it for the second.

## Benchmark

| Property | Value |
|---|---|
| Images | 200 synthetic images, 320 x 320, seed 2026 |
| Scene groups | 40, each with base, spatial, brightness, clutter and occlusion variants |
| Image pairs | 160 (40 spatial-change, 120 label-preserving) |
| Relations (groups) | left 8, right 8, above 6, below 6, inside 6, outside 6 |
| Labels | 100 yes, 100 no |
| Dataset SHA-256 | `c3893bc2d55ca40cc5ba2dbaeb5cc1defb1b6ea568f47771b7d68e328d4dd232` |

All six models were run on these same files.

## Models

| Key | Checkpoint | Revision | Run on |
|---|---|---|---|
| `smol500` | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` | `4cde9d208e19519c36917aa170fffc7e381aa0ad` | GPU |
| `gitvqa` | `microsoft/git-base-vqav2` | `a1847946b7f6bd5aa92d1326714d50dd7b11aad0` | GPU |
| `qwen2vl` | `Qwen/Qwen2-VL-2B-Instruct` | `895c3a49bc3fa70a340399125c650a463535e71c` | GPU |
| `blip` | `Salesforce/blip-vqa-base` | `787b3d35d57e49572baabd22884b3d5a05acf072` | CPU |
| `internvl` | `OpenGVLab/InternVL3-1B-hf` | `014c0583a0d4bedf29fbe2dbff4f865eb998e171` | CPU |
| `gemma` | `google/gemma-3-4b-it` | `093f9f388b31de276ce2de164bdc2081324b9767` | CPU |

Model weights are not included. They download from Hugging Face under their own licenses.

## Results

All values are percentages.

| Model | Image accuracy | Yes recall | No recall | Spatial sensitivity | Prediction invariance | Invariant-pair accuracy |
|---|---:|---:|---:|---:|---:|---:|
| SmolVLM2-500M | 54.5 | 99 | 10 | 5.0 | 94.2 | 52.5 |
| GIT-base-VQAv2 | 49.0 | 98 | 0 | 0.0 | 98.3 | 48.3 |
| Qwen2-VL-2B | **97.0** | 97 | 97 | **97.5** | 95.0 | **94.2** |
| BLIP VQA Base | 52.5 | 99 | 6 | 7.5 | 99.2 | 51.7 |
| InternVL3-1B | 67.0 | 100 | 34 | 40.0 | 93.3 | 61.7 |
| Gemma 3 4B | 63.0 | 100 | 26 | 35.0 | 92.5 | 57.5 |

High prediction invariance is not evidence of robustness on its own: a model that always answers
"yes" is perfectly invariant. Read it together with no recall and spatial sensitivity.

## Repository structure

```
.
├── code/
│   ├── gpu_runs/        Code used for SmolVLM2, GIT and Qwen2-VL, plus dataset generation
│   └── cpu_runs/        Code used for BLIP, InternVL3 and Gemma 3
├── data/final_200/      Images, manifest, pair definitions and dataset hash
├── results/
│   ├── runs/            Per model: predictions, run metadata, package list and metrics
│   └── summary/         Cross-model comparison tables and plots
├── report_tools/        Scripts that check reported numbers and prepare report figures
├── figures/             Report figures (source/ holds the figure inputs)
├── docs/ai_prompts.md   Prompts given to AI tools
└── requirements.txt
```

The main scripts in `code/` are:

| Script | Purpose |
|---|---|
| `generate_200_images.py`, `generate_dataset.py` | Build the benchmark |
| `model_adapters.py` | Load each model, build its prompt and decode its answer |
| `run_model.py` | Run one model and save its predictions |
| `run_all.py` | Run several models in sequence, then analyze them |
| `analyze_results.py` | Compute metrics and bootstrap confidence intervals |
| `common.py` | Dataset validation, hashing and answer parsing |
| `compare_runs.py`, `report_runs.py` | Combine runs into comparison tables and plots |

## Installation

The runs used Python 3.11. Create an environment, install PyTorch for your device, then the rest.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows (Git Bash): source .venv/Scripts/activate

# choose one
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126   # NVIDIA GPU
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu     # CPU only

pip install -r requirements.txt
```

## Reproducing the study

Run every command from the repository root in a Bash shell (on Windows, use Git Bash).
Steps 1 to 3 need no model downloads or GPU.

### 1. Check the reported numbers from the saved predictions

Keep the model order shown below: the scripts assign model names by position.

```bash
RUNS="results/runs/smol500_final200* results/runs/gitvqa_final200* results/runs/qwen2vl_final200* \
      results/runs/blip_final200* results/runs/internvl_final200* results/runs/gemma_final200*"

python report_tools/verify.py $RUNS
python report_tools/make_result_figures.py figures $RUNS
```

The other scripts in `report_tools/` prepare the architecture, relation and benchmark figures from
`figures/source/`. `crop_and_fix.py` also needs the Inter font.

### 2. Recompute metrics and confidence intervals

```bash
for run in results/runs/*; do
  case $run in *_cuda_*) code=code/gpu_runs ;; *) code=code/cpu_runs ;; esac
  python $code/analyze_results.py --manifest data/final_200/manifest.jsonl \
    --predictions $run/predictions.jsonl --out results/recomputed/$(basename $run)
done

# combine the six saved analyses into one table
python code/gpu_runs/compare_runs.py --analyses results/runs/*/analysis_* \
  --out results/recomputed_comparison.csv
```

With the default settings, the recomputed values and intervals match the saved `metrics.csv` files.
The CPU analysis script does not write yes or no recall; `verify.py` in step 1 prints the confusion
counts for all six models.

### 3. Regenerate the dataset (optional)

```bash
python code/gpu_runs/generate_200_images.py --out data/regenerated
```

The seed fixes the scenes, labels and pairs. PNG compression can differ between Pillow builds, so a
regenerated set may have a different SHA-256 even when its pixels match. Use `data/final_200` to
reproduce the reported runs.

### 4. Rerun inference

Each command pins the model revision used in the study. GPU models:

```bash
python code/gpu_runs/run_model.py --model qwen2vl \
  --revision 895c3a49bc3fa70a340399125c650a463535e71c \
  --manifest data/final_200/manifest.jsonl --groups 0 \
  --device cuda --dtype bfloat16 --threads 4 --max-new-tokens 8 \
  --out results/new/qwen2vl/predictions.jsonl

python code/gpu_runs/analyze_results.py --manifest data/final_200/manifest.jsonl \
  --predictions results/new/qwen2vl/predictions.jsonl --out results/new/qwen2vl/analysis
```

For `smol500` and `gitvqa`, change the model key, revision and output folder.

CPU models:

```bash
python code/cpu_runs/run_model.py --model blip \
  --revision 787b3d35d57e49572baabd22884b3d5a05acf072 \
  --manifest data/final_200/manifest.jsonl --groups 0 \
  --device cpu --dtype float32 --threads 4 --max-new-tokens 8 \
  --out results/new/blip/predictions.jsonl

python code/cpu_runs/analyze_results.py --manifest data/final_200/manifest.jsonl \
  --predictions results/new/blip/predictions.jsonl --out results/new/blip/analysis
```

For `internvl` and `gemma`, change the model key, revision and output folder.
Gemma 3 is gated: accept its terms on Hugging Face and run `huggingface-cli login` first.

An interrupted run continues if you repeat the command with `--resume`.

## Evaluation protocol

| Setting | Value |
|---|---|
| Prompt: SmolVLM2, Qwen2-VL, InternVL3, Gemma 3 | Question, then `Answer with exactly one word: yes or no.` |
| Prompt: GIT, BLIP | Question only |
| Decoding | Greedy, at most 8 new tokens |
| Image preprocessing | Each checkpoint's default |
| Answer parsing | The answer must start with `yes` or `no` and contain only one of them; anything else counts as wrong |
| Confidence intervals | 95% percentile bootstrap, 1,000 resamples of the 40 scene groups, seed 42 |

| Metric | Definition |
|---|---|
| Image accuracy | Correct answers over all 200 images |
| Spatial sensitivity | Spatial-change pairs where the answer changed |
| Prediction invariance | Label-preserving pairs where the answer stayed the same |
| Invariant-pair accuracy | Label-preserving pairs where both answers were correct |

## Environments

| | GPU runs | CPU runs |
|---|---|---|
| Hardware | NVIDIA GeForce RTX 4090 | AMD processor, 4 threads |
| Precision | bfloat16 | float32 |
| System | Linux, Python 3.11.5, CUDA 12.6 | Windows, Python 3.11.9 |
| PyTorch / Transformers | 2.7.1 / 4.55.4 | 2.7.1 (CPU build) / 4.55.4 |
| Code SHA-256 prefix | `12ef4bc0b36e` | `70004cc081e4` |

## Notes on reproducibility

- **Two code versions.** The CPU runs used an earlier version of the runner that registers fewer
  models. Both versions are included unchanged. Each `predictions.run.json` records the code hash,
  computed over `model_adapters.py`, `run_model.py` and `common.py`, in that order.
- **Exact bytes.** `.gitattributes` turns off line-ending conversion so that the dataset and code
  hashes still match after cloning.
- **Runtime.** GPU and CPU timings come from different hardware and precision, so they are not a
  like-for-like speed comparison.
- **Saved metadata.** Paths inside the cross-model comparison files were shortened to
  repository-relative paths.
- **Scope.** The benchmark is synthetic, so the results should not be generalized to natural images
  without further testing.

## Use of AI tools

See the AI Use Disclosure in the report and the prompts in `docs/ai_prompts.md`.
