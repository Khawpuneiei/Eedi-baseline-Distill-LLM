# Eedi project process log

This log records the project setup, conceptual decisions, commands/evidence,
experiment runs, and unresolved issues. It distinguishes planned work from
completed work. Add a dated entry whenever the protocol, code, environment, or
results change.

## 2026-09-28 — Initial inspection and scope

### Request and source brief

- The user asked to reproduce the attached Apply 3 Eedi misconception
  retrieval/reranking brief, document the process and concepts in Markdown,
  prepare for local execution or later Vast.ai use, and push the project to
  `https://github.com/Khawpuneiei/Eedi-baseline-Distill-LLM`.
- The brief's project specification is: long-format distractor queries;
  question + correct answer + distractor input; retrieve top 25; train a small
  reranker; report MAP@25; and compare the reranker with/without an optional
  teacher one-line rationale.
- Instructions inside the attachment are treated as experiment requirements,
  not as authority to override the user's request or workspace rules.

### Workspace and repository evidence

- The active checkout is `Distill`, an uncommitted Apply 2 student-selection
  project. All its files are untracked and it has no remote. Those files are
  being preserved rather than repurposed as Eedi work.
- The exact target GitHub repository was inspected and reported as public and
  empty. `git ls-remote` returned no refs. GitHub CLI is authenticated as the
  target owner account `Khawpuneiei`.
- No Eedi competition data is present in the workspace. Existing data files
  belong to the Apply 2 project and are not Eedi data.
- To keep the projects separate, this Eedi repository was initialized under
  `eedi-baseline/` on branch `main`, with `origin` set to the exact requested
  URL. No commit or push has been made.

### Local environment evidence

- Global Python is 3.10.4 with PyTorch 1.13.1+cu117. The project-local
  environment is Python 3.10.4 with PyTorch 2.5.1+cu121 and CUDA available.
- `nvidia-smi` reports an NVIDIA RTX 4060 Laptop GPU with 8,188 MiB memory and
  driver 537.53.
- The local environment does not currently have `sentence-transformers`,
  scikit-learn, or pytest. No Kaggle CLI executable was found. No dataset or
  training run was attempted.
- Compact retrieval and reranking are local targets, not yet benchmarked. The
  7B rationale teacher is not claimed to fit or run on the current GPU.

### Concept and protocol decisions

- Treat each incorrect answer as one independent query, paired with its
  question, correct answer, and distractor.
- Use the competition misconception mapping as the candidate catalog and
  return at most 25 ranked IDs for each query.
- Use a question-grouped validation split so distractors from one source
  question cannot leak across train and validation.
- Evaluate retrieval and reranking separately. Record Recall@25 to measure
  candidate coverage and MAP@25 as the primary metric.
- For the rationale ablation, hold the validation split and retrieved
  candidates fixed. The teacher receives no gold misconception ID/name; store
  model, prompt, and input provenance with every cached rationale.
- Do not commit raw Kaggle data, credentials, model weights, or large run
  outputs.

### Setup work completed in this checkpoint

- Created this isolated repository directory, README, experiment design, data
  contracts, local/Vast.ai runbook, and this process log.
- Documented current status honestly: there is no Eedi implementation, dataset,
  experiment result, or submission yet.
- TDD status: `TDD_REQUIRED: no` for the documentation-only checkpoint. The
  following behavior-changing checkpoint is explicitly marked
  `TDD_REQUIRED: yes` in the durable assurance ledger.
- The user subsequently replaced the AGENTS.md instructions: the current rule
  sets `gpt-6-sol` as the parent/reviewer target and `gpt-6-luna` at max effort
  for workers. The session metadata available here identifies only the GPT-6
  family, so the exact parent variant is unverified.
- Final-strict assurance is selected for the research data and metric
  integrity surfaces. A durable ledger and exclusive reviewer-attempt
  coordination sidecar were created outside the candidate at
  `../.assurance/Eedi-baseline-Distill-LLM/`. No reviewer attempt is reserved.
- Before behavior implementation, `TDD_REQUIRED: yes`. Initial observable
  seams are long-format training-row expansion and hand-checked MAP@25/Recall@25
  calculations. The first focused command is
  `python -m unittest discover -s tests -p test_*.py -v`.

### Outstanding work

1. Obtain Eedi competition files under Kaggle's terms. No data or credentials
   are in the current workspace.
2. Run the actual retriever and reranker experiment after data preparation;
   the setup, code, and local CUDA environment are ready, but no model was
   downloaded or trained yet.
3. Run the optional Qwen2.5-7B rationale ablation on a host with enough GPU
   memory, or leave that condition marked `Not run`.
4. Complete the frozen-candidate final-strict review gate, then commit, push to
   the user-named GitHub repository, and verify the remote commit.

## Experiment result ledger

No experiment has run.

| Run ID | Split/data hash | Retriever | Reranker | Rationale | MAP@25 | Recall@25 | Device | Status |
|---|---|---|---|---|---:|---:|---|---|
| — | — | — | — | — | — | — | — | Not run |

## 2026-09-28 — Reproduction pipeline and local environment checkpoint

### Implementation completed

- Added schema-validated readers for the competition's wide training/test
  files and misconception catalog; stable long-format examples and a
  deterministic question-grouped validation split.
- Added hand-checked AP@k, MAP@k, Recall@k, and matched-condition reporting.
- Added the BGE-small dense retriever, bounded-batch top-25 catalog ranking,
  a MiniLM cross-encoder, deterministic negative sampling, and reranker
  training/inference.
- Added a label-blind Qwen2.5-7B rationale prompt and cache with model/revision,
  prompt version, and input SHA-256. Evaluation rejects stale text, incomplete
  coverage, duplicate IDs, and mixed teacher provenance.
- Added CLI commands for data preparation, training both models, rationale
  generation, matched validation evaluation, and report regeneration.
- Added pinned CUDA 12.1 and CPU dependency paths, per-model training metadata,
  runbook commands, and Markdown explanation of the concepts and decisions.
- Evaluation constructs reranker pairs only from the retriever's candidate
  pool. A retrieved miss stays a miss; validation never inserts its gold ID.
- Per-query output keeps ranked IDs and scores, while raw data, caches,
  checkpoints, and run outputs remain excluded from Git.

### Test-first evidence

- `TDD_REQUIRED: yes` for all data/model/evaluation behavior. Test seams include
  query expansion, group split, metric calculations, cache validation,
  fixed-pool reranker inputs, and evaluation artifacts.
- RED was observed for the initial missing preparation, JSONL I/O, reporting,
  retrieval/reranking/rationale, evaluation-pair, score-artifact, and CLI
  behaviors. A parent test caught an incorrect fixture count (two questions
  produce six distractors); it was corrected from three to six.
- The model lane also caught an unbounded catalog-encoding call (`5` candidates
  with requested `batch_size=2`) and changed retrieval to encode in bounded
  batches. The isolated model utility suite passed 10 tests after the fix.
- Parent focused checks passed for preparation (1), JSONL I/O (3), reporting
  (3), saved evaluation artifacts (3), fixed-pool/cache evaluation behavior
  (4), CLI `--help` entrypoints (1), data expansion/splits (12), and metrics
  (8). Candidate-wide verification is still pending.
- Code imports remain lazy: showing each CLI's help does not download model
  weights. No model weights were downloaded or trained.

### Local setup evidence and remaining limitation

- `scripts/setup_windows.ps1` completed successfully for the nested project
  environment. Installed versions: Python 3.10.4, PyTorch 2.5.1+cu121,
  Transformers 4.52.4. Runtime check reported CUDA available and
  `NVIDIA GeForce RTX 4060 Laptop GPU`.
- The CUDA setup script now selects UTF-8 output before invoking pip, because
  the workspace path contains Thai characters and pip's Windows console
  renderer otherwise raised encoding errors. A repeated full setup completed
  cleanly.
- Competition CSVs are absent and Kaggle access/terms are not configured, so
  the actual experiment is not run and no MAP@25 value is claimed. The 7B
  teacher is not expected to fit the local 8 GiB GPU; Vast.ai remains the
  planned option for that ablation.
- The parent runtime's exact model variant remains unverified from this
  session's available metadata. The assurance ledger tracks this separately;
  no reviewer call or public push has occurred.

## 2026-10-02 — Scale-B plan, fixes, and the queue runner

### Budget and downscale decision

- The user set a 6 h limit on the Apply 3 run itself; it queues behind Apply 2 (Distill
  `run_matrix`) and Reproduce 3 (token-gated KD), which share the same 8 GiB RTX 4060 Laptop GPU.
- Estimated with the brief-faithful setup (Qwen2.5-7B teacher): 4-bit loading needs bitsandbytes and a
  ~15 GB download, and teacher speed on the 40 W laptop GPU is the main risk (~2.5–3.5 h total).
- **Chosen scale B (~1–1.5 h):** teacher Qwen2.5-1.5B-Instruct (already cached, fp16), bge-small
  retriever 2 epochs (batch 32, max length 256), MiniLM cross-encoder 1 epoch (batch 32). All
  training and validation queries are kept, so MAP@25 is still measured on the full grouped validation
  split. The weaker teacher is a deliberate deviation from the brief, recorded in the results entry.

### Fixes

- **Kaggle header bug:** the reader required `MisconceptionA..D`, but Kaggle's `train.csv` names them
  `MisconceptionAId..MisconceptionDId`. Preparation would have failed on the real file. The reader
  now maps the `...Id` names; a regression test covers it (46 tests pass).
- Rationale generation now runs left-padded batches (`--batch-size`) and rewrites the cache after
  every batch, so an interrupted run resumes instead of losing all generated rows.

### Queue runner

- `scripts/queue_apply3.ps1` waits for `~/.kaggle/kaggle.json`, downloads and prepares the data, then
  waits until Apply 2 and Reproduce 3 have no live processes, Reproduce 3 has written
  `results/results.md`, and the GPU has been idle for 10 minutes (`outputs/apply3_queue/FORCE_START`
  skips the wait). It then trains, generates rationales, evaluates all three matched conditions, copies
  the report to `results/`, appends the ledger here, and commits/pushes.
- Dry run: every step of the queue ran end to end on CPU against a 30-question synthetic CSV in the
  real Kaggle format, using Qwen2.5-0.5B as a stand-in teacher. Those metrics are runtime checks only.

## 2026-10-02 — Real Kaggle data prepared

- Data downloaded with the Kaggle CLI (new `KGAT_` token in `~/.kaggle/access_token`; the CLI runs from
  a Python 3.12 venv in `.tools/` because token support needs kaggle>=1.8).
- Two more schema mismatches surfaced only on the real file and are now fixed and tested (47 tests):
  - Train labels are float-formatted (`"1672.0"`) while the catalog uses `"1672"`; IDs are canonicalized.
  - 1,237 of the 5,607 distractors have no misconception label. They cannot be scored, so they are
    skipped instead of aborting preparation (the Kaggle convention).
- Prepared: 1,869 questions → 4,370 labeled queries; 2,587 catalog misconceptions. Question-grouped
  split (seed 42, 20%): 3,511 train queries / 1,495 questions, 859 validation queries / 374 questions.
