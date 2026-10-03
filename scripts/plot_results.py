"""Draw the README figures from results/metrics.json and results/analysis.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CONDITIONS = ("retriever", "reranker", "rationale")
NAME = {"retriever": "Retriever", "reranker": "+ Reranker", "rationale": "+ Rationale"}
COLOR = {"retriever": "#2a78d6", "reranker": "#eb6834", "rationale": "#1baf7a"}
INK, INK2, GRID, NEUTRAL = "#14181f", "#4a5160", "#e3e7ee", "#d6d9df"
RANK_COLORS = ("#104281", "#256abf", "#5598e7", "#86b6ef", NEUTRAL)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, default=Path("results/analysis.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures"))
    return parser


def style() -> None:
    plt.rcParams.update({
        "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
        "ytick.color": INK2, "axes.titleweight": "bold", "axes.titlesize": 12, "axes.titlelocation": "left",
        "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 150, "savefig.bbox": "tight",
    })


def forest(analysis: dict, path: Path) -> None:
    rows = [
        ("All queries: reranker vs retriever", "all", "reranker-retriever", "reranker"),
        ("All queries: rationale vs reranker", "all", "rationale-reranker", "rationale"),
        ("Seen in training: reranker vs retriever", "seen", "reranker-retriever", "reranker"),
        ("Seen in training: rationale vs reranker", "seen", "rationale-reranker", "rationale"),
        ("Never seen: reranker vs retriever", "unseen", "reranker-retriever", "reranker"),
        ("Never seen: rationale vs reranker", "unseen", "rationale-reranker", "rationale"),
    ]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for i, (label, group, key, condition) in enumerate(rows):
        stats = analysis["groups"][group]["deltas"][key]
        y = len(rows) - 1 - i
        lo, hi = stats["ci95"]
        ax.plot([lo, hi], [y, y], color=COLOR[condition], lw=2, solid_capstyle="round")
        ax.plot([lo, lo], [y - 0.15, y + 0.15], color=COLOR[condition], lw=2)
        ax.plot([hi, hi], [y - 0.15, y + 0.15], color=COLOR[condition], lw=2)
        ax.scatter([stats["mean"]], [y], s=55, color=COLOR[condition], edgecolor="white", linewidth=1.5, zorder=3)
        ax.text(0.125, y, f"{stats['mean']:+.3f}", va="center", ha="left", color=INK, fontsize=9)
    ax.axvline(0, color=INK2, lw=1, ls=(0, (3, 3)))
    ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)])
    ax.set_xlim(-0.12, 0.12)
    ax.set_xlabel("Change in MAP@25 (95% interval, questions resampled)")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Reranking helps on seen misconceptions and hurts on unseen ones")
    fig.savefig(path)
    plt.close(fig)


def recall_curve(analysis: dict, path: Path) -> None:
    k = analysis["k"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for condition in CONDITIONS:
        curve = analysis["recall_curve"][condition]
        ax.plot(range(1, k + 1), curve, color=COLOR[condition], lw=2,
                label=f"{NAME[condition]} (top-1 {curve[0]:.1%})")
    final = analysis["recall_curve"]["retriever"][-1]
    ax.annotate(f"{final:.1%} for all three:\nthe retriever's ceiling", xy=(k, final), xytext=(k - 7.5, final - 0.2),
                fontsize=9, color=INK2, arrowprops={"arrowstyle": "-", "color": INK2, "lw": 0.8})
    ax.set_xlim(1, k)
    ax.set_ylim(0, 0.75)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xticks([1, 5, 10, 15, 20, 25])
    ax.set_xlabel("Cutoff k")
    ax.set_ylabel("Correct misconception in top k")
    ax.grid(color=GRID, lw=0.8)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Recall@k on 859 validation queries")
    fig.savefig(path)
    plt.close(fig)


def rank_bins(analysis: dict, path: Path) -> None:
    bins = ("1", "2-5", "6-10", "11-25", "not_retrieved")
    labels = ("Rank 1", "Rank 2–5", "Rank 6–10", "Rank 11–25", "Not in top 25")
    total = analysis["queries"]
    fig, ax = plt.subplots(figsize=(8, 2.6))
    for i, condition in enumerate(CONDITIONS):
        y = len(CONDITIONS) - 1 - i
        left = 0
        for b, color, label in zip(bins, RANK_COLORS, labels):
            n = analysis["rank_bins"][condition][b]
            ax.barh(y, n - 2, left=left, height=0.62, color=color, label=label if i == 0 else None)
            if n > 45:
                ax.text(left + n / 2, y, str(n), ha="center", va="center", fontsize=9,
                        color="white" if color in RANK_COLORS[:2] else INK)
            left += n
    ax.set_yticks(range(len(CONDITIONS)), [NAME[c] for c in reversed(CONDITIONS)])
    ax.set_xlim(0, total)
    ax.set_xlabel("Validation queries")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.legend(frameon=False, ncol=5, loc="upper left", bbox_to_anchor=(0, -0.32), fontsize=9)
    ax.set_title("Where the correct misconception lands")
    fig.savefig(path)
    plt.close(fig)


def seen_unseen(analysis: dict, path: Path) -> None:
    groups = (("seen", "Seen in training"), ("unseen", "Never seen"))
    fig, ax = plt.subplots(figsize=(8, 3.4))
    width = 0.24
    for gi, (group, label) in enumerate(groups):
        for ci, condition in enumerate(CONDITIONS):
            value = analysis["groups"][group]["map_at_k"][condition]
            x = gi + (ci - 1) * (width + 0.02)
            ax.bar(x, value, width=width, color=COLOR[condition], label=NAME[condition] if gi == 0 else None)
            ax.text(x, value + 0.006, f"{value:.3f}", ha="center", fontsize=9, color=INK)
    ax.set_xticks([0, 1], [f"{label}\n{analysis['groups'][g]['queries']} queries" for g, label in groups])
    ax.set_ylim(0, 0.35)
    ax.set_ylabel("MAP@25")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)
    ax.legend(frameon=False, ncol=3, loc="upper right")
    ax.set_title("MAP@25 by whether the misconception appears in training")
    fig.savefig(path)
    plt.close(fig)


def per_query(analysis: dict, path: Path) -> None:
    rows = (("reranker-retriever", "Reranker vs retriever"), ("rationale-reranker", "Rationale vs reranker"))
    parts = (("higher", "#2a78d6", "Ranked higher"), ("same", NEUTRAL, "Same rank"), ("lower", "#e34948", "Ranked lower"))
    fig, ax = plt.subplots(figsize=(8, 2.1))
    for i, (key, label) in enumerate(rows):
        y = len(rows) - 1 - i
        left = 0
        for part, color, part_label in parts:
            n = analysis["per_query"][key].get(part, 0)
            ax.barh(y, n - 2, left=left, height=0.6, color=color, label=part_label if i == 0 else None)
            ax.text(left + n / 2, y, str(n), ha="center", va="center", fontsize=9, color="white" if part != "same" else INK)
            left += n
    ax.set_yticks(range(len(rows)), [r[1] for r in reversed(rows)])
    ax.set_xlim(0, analysis["queries"])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Validation queries (same rank includes the 266 neither model retrieves)")
    ax.legend(frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0, -0.42), fontsize=9)
    ax.set_title("Query by query: a near-zero mean hides swaps in both directions")
    fig.savefig(path)
    plt.close(fig)


def topics(analysis: dict, path: Path) -> None:
    rows = analysis["largest_subjects"]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for i, row in enumerate(rows):
        y = len(rows) - 1 - i
        values = [row[c] for c in CONDITIONS]
        ax.plot([min(values), max(values)], [y, y], color=NEUTRAL, lw=2, zorder=1)
        for condition in CONDITIONS:
            ax.scatter(row[condition], y, s=50, color=COLOR[condition], edgecolor="white", linewidth=1.2, zorder=2,
                       label=NAME[condition] if i == 0 else None)
    ax.set_yticks(range(len(rows)), [f"{r['subject']} (n={r['queries']})" for r in reversed(rows)])
    ax.set_xlim(0, 0.65)
    ax.set_xlabel("MAP@25 (small groups: a rough picture only)")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0, -0.16), fontsize=9)
    ax.set_title(f"Largest 8 of {analysis['subjects_total']} topics")
    fig.savefig(path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    style()
    for name, draw in (("gains", forest), ("recall_curve", recall_curve), ("rank_bins", rank_bins),
                       ("seen_unseen", seen_unseen), ("per_query", per_query), ("topics", topics)):
        path = args.output_dir / f"{name}.png"
        draw(analysis, path)
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
