# Experiment design

## Question and prediction unit

The project follows the supplied Apply 3 brief: can adding a short teacher
rationale about a student's likely error improve misconception retrieval as
measured by MAP@25?

Each incorrect answer is one prediction unit. Its query text contains the math
question, correct answer, and that one distractor. Its target is the
misconception ID attached to that incorrect answer. The misconception mapping
is the candidate catalog. Test-time candidate output is limited to the top 25
IDs.

## Conditions

| Condition | Candidate generation | Reranker input |
|---|---|---|
| A: retriever | Dense cosine ranking over the catalog | None |
| B: reranker | Same retriever top-25 pool | Query and candidate misconception description |
| C: rationale reranker | Same retriever top-25 pool | Query, candidate description, and label-blind teacher rationale |

Condition C is optional. If teacher compute is not available, the report marks
it `Not run`; it never invents a score.

## Implementation choices

### Data preparation and split

The preparation command validates the wide Kaggle schema, checks every
training label against the catalog, and expands each source row into one record
per incorrect answer. It does not label the correct answer as a negative
example. The stable query key is `QuestionId_AnswerLetter`.

The deterministic 80/20 default split groups by source `QuestionId`. All
distractors from one question stay in the same split. The manifest stores the
source CSV hashes, seed, split fraction, and row/question counts.

### Dense retrieval

The default encoder is `BAAI/bge-small-en-v1.5`; the model ID is configurable.
The query encoder uses BGE's short-query prefix, and the candidate text is the
misconception description. Training uses a batch multi-positive contrastive
objective: repeated misconception labels in a batch are all marked positive,
so they do not become false negatives. The implementation saves its base model
revision, query prefix, max length, and a separate training-run record.

Inference embeds the full catalog in bounded batches, normalizes vectors, and
ranks by cosine similarity. Ties use misconception ID order. It saves at most
25 candidates per query. Recall@25 shows whether the gold label survived this
candidate-generation step.

### Cross-encoder reranker

The default model is `cross-encoder/ms-marco-MiniLM-L-6-v2`. Training creates a
positive pair for the gold label, includes retrieved non-gold candidates, and
adds a fixed-seed sample of random negatives. The gold positive is included in
training even if retrieval missed it; validation never inserts the gold label
into the candidate pool.

The reranker uses a one-logit binary relevance objective. During evaluation,
each query-candidate pair is formed only from the already retrieved pool. The
reranker reorders those IDs and cannot add a candidate that retrieval omitted.
Evaluation checks exact query coverage for every condition before writing any
artifact.

### Rationale ablation

The optional teacher defaults to `Qwen/Qwen2.5-7B-Instruct`. Its prompt uses
only question text, correct answer text, and distractor text. It explicitly
does not request a misconception category and never reads the gold ID or name.
Generation is deterministic and limited to one short continuation.

Each cache row records the rationale, teacher model/revision, prompt version,
query ID, and SHA-256 of the complete prompt. Evaluation recomputes that hash
from the current validation row and rejects stale or incomplete caches. Train
and validation rationales must be generated from their respective splits.

### Metric

MAP@25 is the primary metric. Average precision uses the ranked list truncated
to 25 and divides by `min(25, number of relevant IDs)`. The Eedi long-format
records currently have one gold ID per distractor; a correct ID at rank `r`
contributes `1/r`, and a missing ID contributes zero. Recall@25 is reported as
a retrieval diagnostic.

Unit tests check the metric with hand-derived values, including rank 25,
multiple relevant IDs, duplicate predictions, and misses. The evaluation code
uses the same validation rows and the same retrieved candidate lists for all
conditions. Prediction files preserve candidate scores and IDs for error
analysis.

## Reproducibility and limits

- The split seed, input hashes, model IDs/revisions, training settings,
  prediction files, and runtime metadata are saved with local run artifacts.
- Raw competition files, rationale caches, checkpoints, and outputs stay out
  of Git.
- The local RTX 4060 Laptop has 8 GiB VRAM. Batch sizes default to 8 for model
  training, but no actual model-training throughput or memory benchmark has
  been measured.
- The optional 7B rationale teacher is not expected to fit in 8 GiB VRAM in
  full precision. Use a larger Vast.ai GPU or defer the condition; no result
  should be reported until it has actually run.
- The project has no competition data in the repository. Download it only
  after obtaining Kaggle competition access under its terms.

## Result table

No Eedi training or validation experiment has run. Fill this table only from
saved evaluation outputs.

| Condition | MAP@25 | Recall@25 | Validation queries | Status |
|---|---:|---:|---:|---|
| Retriever | — | — | — | Not run |
| Retriever + reranker | — | — | — | Not run |
| Retriever + rationale reranker | — | — | — | Not run |

## Source

- User-supplied `apply-3-eedi-misconception.md` brief.
- [Official Kaggle competition overview](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/overview).
- [BGE-small-en-v1.5 model card](https://huggingface.co/BAAI/bge-small-en-v1.5) for the recommended short-query prefix.
