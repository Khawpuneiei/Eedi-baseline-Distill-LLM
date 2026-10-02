# Local and Vast.ai runbook

## Machine and expected limits

The inspected Windows machine has Python 3.10.4, an NVIDIA RTX 4060 Laptop
GPU, and 8 GiB reported VRAM. The compact BGE retriever and MiniLM cross
encoder are configured with batch size 8 as a starting point. They have not
been trained on Eedi data on this machine, so runtime and memory claims remain
unmeasured.

The optional Qwen2.5-7B rationale teacher is not included in the local
feasibility claim. Its full-precision weights exceed the local GPU memory.
Generate rationales on a larger GPU or defer that ablation.

## Environment setup

Windows with the observed CUDA 12.1 setup:

```powershell
.\scripts\setup_windows.ps1
```

Linux with CUDA 12.1:

```bash
bash scripts/setup_linux.sh
```

CPU-only setup:

```bash
python -m venv .venv
python -m pip install -r requirements-cpu.txt
python -m pip install --no-deps -e .
```

Check the environment before training:

```bash
python --version
python -c "import torch, transformers; print('torch', torch.__version__); print('transformers', transformers.__version__); print('CUDA', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Use the virtual environment's Python for the remaining commands.

## Data setup

1. Open the [official Eedi competition](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics), sign in, and obtain data under its terms.
2. Put `train.csv`, `test.csv`, and `misconception_mapping.csv` in `data/raw/`.
3. Prepare the long-format files and deterministic question-grouped split:

   ```bash
   python -m scripts.prepare_data \
     --train data/raw/train.csv \
     --test data/raw/test.csv \
     --mapping data/raw/misconception_mapping.csv \
     --output-dir data/processed --validation-fraction 0.2 --seed 42
   ```

4. Keep `data/processed/manifest.json`; it binds the exact source hashes and
   split settings.

No Kaggle CLI or competition data was found in the initial machine inspection.
Do not assume that credentials or data-use terms have been configured.

## Baseline training and evaluation

```bash
python -m scripts.train_retriever \
  --train data/processed/train.jsonl \
  --output-dir models/retriever

python -m scripts.train_reranker \
  --train data/processed/train.jsonl \
  --catalog data/processed/catalog.json \
  --retriever models/retriever \
  --output-dir models/reranker

python -m scripts.evaluate \
  --validation data/processed/validation.jsonl \
  --catalog data/processed/catalog.json \
  --retriever models/retriever \
  --reranker models/reranker \
  --output-dir outputs/validation
```

Each training directory contains model configuration plus `training_run.json`
with data hashes, hyperparameters, model revision, runtime version, and device
information. Evaluation saves per-query rankings/scores, metrics, and a
Markdown report. The reranker uses the exact retriever candidate pool.

## Rationale ablation

Generate separate training and validation caches. The teacher call never sees
gold misconception IDs or descriptions. Use a larger GPU if the 7B model does
not fit:

```bash
python -m scripts.generate_rationales \
  --input data/processed/train.jsonl \
  --output data/cache/rationales_train.jsonl

python -m scripts.generate_rationales \
  --input data/processed/validation.jsonl \
  --output data/cache/rationales_validation.jsonl

python -m scripts.train_reranker \
  --train data/processed/train.jsonl \
  --catalog data/processed/catalog.json \
  --retriever models/retriever \
  --rationales data/cache/rationales_train.jsonl \
  --output-dir models/reranker_rationale

python -m scripts.evaluate \
  --validation data/processed/validation.jsonl \
  --catalog data/processed/catalog.json \
  --retriever models/retriever \
  --reranker models/reranker \
  --rationale-reranker models/reranker_rationale \
  --rationales data/cache/rationales_validation.jsonl \
  --output-dir outputs/validation_with_rationale
```

Copy only code, configs, and the prepared split manifest between local and
Vast.ai. Use the same source data hashes, split seed, model revisions,
hyperparameters, and candidate depth. Keep downloaded data, caches, model
weights, and full prediction outputs out of Git.

## Checks

Run the behavior suite and static compilation:

```bash
python -m unittest discover -s tests -v
python -m compileall -q eedi_baseline scripts
```

These checks need no model downloads. A successful check does not substitute
for a real training/evaluation run.
