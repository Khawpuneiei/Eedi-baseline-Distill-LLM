"""Break down saved validation predictions: rank bins, recall curve, seen/unseen labels,
per-query wins/losses, topics, and question-level paired-bootstrap intervals."""

from __future__ import annotations

import argparse
import collections
import csv
import json
import random
from pathlib import Path

from eedi_baseline.io import read_jsonl

CONDITIONS = ("retriever", "reranker", "rationale")
COMPARISONS = (("reranker", "retriever"), ("rationale", "reranker"), ("rationale", "retriever"))
RANK_BINS = (("1", 1, 1), ("2-5", 2, 5), ("6-10", 6, 10), ("11-25", 11, 25))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-dir", type=Path, default=Path("outputs/validation_with_rationale"))
    parser.add_argument("--validation", type=Path, default=Path("data/processed/validation.jsonl"))
    parser.add_argument("--train", type=Path, default=Path("data/processed/train.jsonl"))
    parser.add_argument("--raw-train", type=Path, default=Path("data/raw/train.csv"))
    parser.add_argument("--rationales", type=Path, default=Path("data/cache/rationales_validation.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/analysis.json"))
    parser.add_argument("--k", type=int, default=25)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--include-text", action="store_true",
        help="Quote competition question/answer/label text in examples (keep such files out of Git)",
    )
    return parser


def question_bootstrap(query_ids, question_of, delta, resamples, seed):
    """Mean of per-query deltas with a 95% interval from resampling whole questions."""
    by_question = collections.defaultdict(list)
    for query_id in query_ids:
        by_question[question_of[query_id]].append(query_id)
    questions = sorted(by_question)
    rng = random.Random(seed)

    def mean(sample):
        ids = [query_id for question in sample for query_id in by_question[question]]
        return sum(delta[query_id] for query_id in ids) / len(ids)

    draws = sorted(mean([rng.choice(questions) for _ in questions]) for _ in range(resamples))
    lower = draws[int(0.025 * resamples)]
    upper = draws[int(0.975 * resamples) - 1]
    return {"mean": mean(questions), "ci95": [lower, upper], "questions": len(questions), "queries": len(query_ids)}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validation = {row["query_id"]: row for row in read_jsonl(args.validation)}
    query_ids = sorted(validation)
    question_of = {query_id: str(row["question_id"]) for query_id, row in validation.items()}
    gold = {query_id: str(row["misconception_id"]) for query_id, row in validation.items()}
    train_counts = collections.Counter(str(row["misconception_id"]) for row in read_jsonl(args.train))

    rank: dict[str, dict[str, int | None]] = {}
    for condition in CONDITIONS:
        rows = {row["query_id"]: row for row in read_jsonl(args.predictions_dir / f"predictions_{condition}.jsonl")}
        if set(rows) != set(query_ids):
            raise ValueError(f"predictions_{condition}.jsonl does not cover the validation queries")
        rank[condition] = {}
        for query_id in query_ids:
            ranked = [str(item) for item in rows[query_id]["ranked_misconception_ids"][: args.k]]
            rank[condition][query_id] = ranked.index(gold[query_id]) + 1 if gold[query_id] in ranked else None

    def ap(condition, query_id):
        value = rank[condition][query_id]
        return 1.0 / value if value else 0.0

    def map_of(condition, ids):
        return sum(ap(condition, query_id) for query_id in ids) / len(ids)

    groups = {
        "all": query_ids,
        "seen": [q for q in query_ids if train_counts[gold[q]] > 0],
        "unseen": [q for q in query_ids if train_counts[gold[q]] == 0],
    }
    analysis: dict = {
        "k": args.k,
        "queries": len(query_ids),
        "questions": len(set(question_of.values())),
        "train_unique_misconceptions": len(train_counts),
        "validation_unique_misconceptions": len(set(gold.values())),
        "bootstrap": {"unit": "question", "resamples": args.resamples, "seed": args.seed},
        "groups": {},
    }
    for name, ids in groups.items():
        delta_stats = {}
        for later, earlier in COMPARISONS:
            delta = {q: ap(later, q) - ap(earlier, q) for q in ids}
            delta_stats[f"{later}-{earlier}"] = question_bootstrap(
                ids, question_of, delta, args.resamples, args.seed
            )
        analysis["groups"][name] = {
            "queries": len(ids),
            "map_at_k": {c: map_of(c, ids) for c in CONDITIONS},
            "recall_at_k": {c: sum(1 for q in ids if rank[c][q]) / len(ids) for c in CONDITIONS},
            "deltas": delta_stats,
        }

    analysis["recall_curve"] = {
        c: [sum(1 for q in query_ids if rank[c][q] and rank[c][q] <= n) / len(query_ids) for n in range(1, args.k + 1)]
        for c in CONDITIONS
    }
    analysis["rank_bins"] = {
        c: {label: sum(1 for q in query_ids if rank[c][q] and lo <= rank[c][q] <= hi) for label, lo, hi in RANK_BINS}
        | {"not_retrieved": sum(1 for q in query_ids if not rank[c][q])}
        for c in CONDITIONS
    }

    def wins_losses(later, earlier):
        counts = collections.Counter()
        for q in query_ids:
            diff = ap(later, q) - ap(earlier, q)
            counts["higher" if diff > 1e-12 else "lower" if diff < -1e-12 else "same"] += 1
        return dict(counts)

    analysis["per_query"] = {f"{later}-{earlier}": wins_losses(later, earlier) for later, earlier in COMPARISONS[:2]}

    with args.raw_train.open(encoding="utf-8-sig", newline="") as handle:
        subject_of = {row["QuestionId"]: row["SubjectName"] for row in csv.DictReader(handle)}
    by_subject = collections.defaultdict(list)
    for q in query_ids:
        by_subject[subject_of[question_of[q]]].append(q)
    analysis["subjects_total"] = len(by_subject)
    analysis["largest_subjects"] = [
        {"subject": subject, "queries": len(ids)} | {c: map_of(c, ids) for c in CONDITIONS}
        for subject, ids in sorted(by_subject.items(), key=lambda item: (-len(item[1]), item[0]))[:8]
    ]

    rationales = {row["query_id"]: row["rationale"] for row in read_jsonl(args.rationales)}
    order = sorted(query_ids, key=lambda q: (ap("rationale", q) - ap("reranker", q), q))

    def example(q):
        record = {"query_id": q, "rationale": rationales[q], "ranks": {c: rank[c][q] for c in CONDITIONS}}
        if args.include_text:
            row = validation[q]
            record |= {
                "question": row["question"], "correct_answer": row["correct_answer"],
                "distractor": row["distractor"], "gold_misconception": row["misconception_name"],
            }
        return record

    analysis["examples"] = {
        "rationale_helped": [example(q) for q in reversed(order[-3:])],
        "rationale_hurt": [example(q) for q in order[:2]],
    }
    analysis["mean_rationale_words"] = sum(len(rationales[q].split()) for q in query_ids) / len(query_ids)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, group in analysis["groups"].items():
        print(f"{name} ({group['queries']} queries)")
        for key, stats in group["deltas"].items():
            lo, hi = stats["ci95"]
            print(f"  {key:22s} {stats['mean']:+.4f}  [{lo:+.4f}, {hi:+.4f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
