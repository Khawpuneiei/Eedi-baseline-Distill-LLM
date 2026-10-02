# Data and artifact contracts

## Source files

Obtain competition data from Kaggle after getting access under the competition
terms. Keep the files under `data/raw/`; the raw-data directory is ignored by
Git. Preparation currently requires these source files:

- `train.csv`: question text, correct answer key, four answer texts, and four
  misconception columns;
- `test.csv`: question text, correct answer key, and four answer texts, with
  no target misconception labels; and
- `misconception_mapping.csv`: unique misconception IDs and descriptions.

The reader checks required headers, duplicate/missing columns, malformed rows,
empty answers, incorrect-answer labels, and unknown catalog IDs. It stops with
a schema/validation error rather than silently dropping an answer.

## Prepared query rows

Each wrong answer becomes one JSON object in `train.jsonl`,
`validation.jsonl`, or `test.jsonl`:

```json
{
  "query_id": "question-id_B",
  "question_id": "question-id",
  "answer_key": "B",
  "question": "Question text",
  "correct_answer": "Correct text",
  "distractor": "Incorrect text",
  "query": "Question: ...\nCorrect answer: ...\nDistractor: ...",
  "misconception_id": "17",
  "misconception_name": "Description"
}
```

The last two fields appear only for labeled training/validation rows. The
preparation manifest records input SHA-256 values, seed, validation fraction,
and row/question counts. The validation split is grouped by `question_id` to
prevent sibling distractors from crossing the split.

## Rationale cache

Each rationale JSONL row includes:

```text
query_id, rationale, teacher_model, teacher_revision, prompt_version, input_sha256
```

The rationale prompt uses only `question`, `correct_answer`, and `distractor`.
It excludes the misconception ID and name. Evaluation validates exact query
coverage, rejects duplicates/unexpected IDs, and recomputes `input_sha256` from
each source row so a cache cannot silently attach to changed text. Keep train
and validation caches separate.

## Evaluation artifacts

The evaluation command writes under `outputs/`:

- `metrics.json` with MAP@25, Recall@25, counts, and run metadata;
- `predictions_<condition>.jsonl` with each query key, gold ID, ordered
  candidate IDs, and candidate scores; and
- `report.md`, regenerated with `python -m scripts.build_report` when needed.

The prepared test set is label-free. The current reproduction flow evaluates
the held-out validation split; it does not create a competition submission.
Do not commit raw data, credentials, rationale caches, model weights, or large
prediction/run outputs. Small aggregate results can be committed only after a
real run and with their data/split provenance.
