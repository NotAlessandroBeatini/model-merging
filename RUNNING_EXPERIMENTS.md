# Running Experiments (Environment + Results)

This doc covers:
- what to put in `.env`
- how to run experiments
- how to read the outputs

## Required `.env`

Create a `.env` file at the repo root with **at least**:

```
MODELS_PATH=/media/focaccia-man/BigDisk/projects/model-merging/data/models
```

Why it’s required:
- `MODELS_PATH` is used by Hydra configs (e.g. `conf/multitask.yaml`) for SVD cache paths, OpenCLIP cache, and model checkpoints.
- If it’s missing, Hydra will fail to resolve `${oc.env:MODELS_PATH}`.

Make sure the directory exists and is writable:

```sh
mkdir -p /media/focaccia-man/BigDisk/projects/model-merging/data/models
```

## Optional `.env`

Add these if you need them:

```
# If you want to avoid WandB login prompts
WANDB_MODE=offline

# Or, if you want online logging
WANDB_API_KEY=your_api_key_here

# If you run into cache permission issues
HF_HOME=/media/focaccia-man/BigDisk/projects/model-merging/data/hf_cache
TRANSFORMERS_CACHE=/media/focaccia-man/BigDisk/projects/model-merging/data/hf_cache
MPLCONFIGDIR=/media/focaccia-man/BigDisk/projects/model-merging/data/mpl_cache
```

## Cluster / Multi-GPU Notes

By default the trainer uses **1 GPU** (`conf/train/default.yaml`):

```
trainer:
  accelerator: 'gpu'
  devices: 1
```

If your node has 4 GPUs and you want to use them:

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 \
conda run -n merge python scripts/evaluate_multitask_merging.py \
  train.trainer.devices=4 \
  train.trainer.strategy=ddp
```

Notes:
- Multi-GPU is not required for correctness; it only helps speed.
- For quick debugging, keep `devices=1`.
- If your cluster disallows outbound internet, pre-download HF models/datasets
  or set `HF_HOME` to a shared cache that already contains them.

## How to run experiments

### 1) Multi-task merging (default evaluation)

```sh
conda run -n merge python scripts/evaluate_multitask_merging.py
```

This uses:
- model: `ViT-B-32` (default in `conf/nn/default.yaml`)
- benchmark: `N20` tasks (default in `conf/multitask.yaml`)
- merger: `dual` (default in `conf/multitask.yaml`)

Override anything via Hydra, for example:

```sh
conda run -n merge python scripts/evaluate_multitask_merging.py \
  nn.encoder=b16 \
  benchmark=N8 \
  merger=isotropic
```

### 2) Tensor-decomposition merger (HOSVD across layers)

```sh
conda run -n merge python scripts/evaluate_multitask_merging.py \
  merger=tensor_decomp \
  merger.rank_strategy=energy \
  merger.energy_task=0.95 \
  merger.energy_layer=0.95 \
  merger.energy_out=1.0 \
  merger.energy_in=1.0
```

Notes:
- The tensor-decomp merger only applies to these layer families:
  - `visual.transformer.resblocks.*.attn.in_proj_weight`
  - `visual.transformer.resblocks.*.attn.out_proj.weight`
  - `visual.transformer.resblocks.*.mlp.c_fc.weight`
  - `visual.transformer.resblocks.*.mlp.c_proj.weight`
- All other parameters fall back to TSV by default.

If you don’t want TSV fallback (and its SVD caching), set:

```sh
conda run -n merge python scripts/evaluate_multitask_merging.py \
  merger=tensor_decomp \
  merger.fallback=mean
```

### 3) Finetuning a single task

```sh
conda run -n merge python scripts/finetune.py dataset=MNIST
```

This writes checkpoints under `checkpoints/` by default (see `conf/finetune.yaml`).

### 4) Minimal quick check (sanity run)

If you want a fast, small run:

```sh
conda run -n merge python scripts/evaluate_multitask_merging.py \
  benchmark=N2 \
  train.trainer.fast_dev_run=true
```

## How to interpret results

### Output files

After `evaluate_multitask_merging.py` finishes, results are written to:

```
results/<model_name>/<num_tasks>.json
```

Example:
```
results/ViT-B-32/20.json
```

### What’s inside

Each JSON contains per-task metrics, plus an `avg` entry. Example keys:
- `acc/test/<dataset_name>`
- `normalized_acc/test/<dataset_name>` (scaled by finetuned upper bounds)
- `acc/test/avg`
- `normalized_acc/test/avg`

### Normalized accuracy

Normalized accuracy is computed using:
```
results/finetuning/accs.json
```

This file provides the “upper bound” accuracy for each dataset and model (used for normalization).

### WandB logging

If WandB is enabled, the script also logs:
- a radar chart comparing tasks
- the full results as an artifact

To disable online logging, set `WANDB_MODE=offline` in `.env`.
