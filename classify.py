"""
Robot Fault Log Classifier

A CLI tool that takes robot fault log descriptions and categorises them
by failure type using a zero-shot natural language inference model.

The model used (facebook/bart-large-mnli) is a pre-trained NLI model from
Hugging Face. Zero-shot classification works by treating each candidate
label as a hypothesis ("This text is about <label>") and scoring how well
the input log entry entails that hypothesis. No fine-tuning is required,
which makes this approach practical for small log sets where labelled
training data is not available.

Motivation
----------
This tool grew out of manual fault analysis work carried out during the
AMRC Coffee-Making Robot project (Jan-May 2026), where runtime issues
including planner failures, IK rejections and collision events were
diagnosed by reading raw logs by hand. Automating the first-pass
categorisation makes it easier to see fault distributions and spot the
dominant failure modes.

Usage
-----
    python classify.py --input logs.csv
    python classify.py --input logs.csv --output results.csv
    python classify.py --input logs.csv --chart fault_breakdown.png

Author: Keegan Chinyenze
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd


# -------- Configuration -------------------------------------------------------

# Candidate categories. Kept deliberately small and non-overlapping so that
# the zero-shot scores are meaningful. Each label is phrased as a short
# natural-language description (this matters - NLI models score the label
# string itself, so descriptive labels work better than terse codes).
CANDIDATE_LABELS = [
    "motion planning failure",
    "collision detection event",
    "inverse kinematics failure",
    "joint or velocity limit violation",
]

# The Hugging Face model name. BART-MNLI is a sensible default for zero-shot:
# pre-trained on MultiNLI, runs on CPU without too much pain, no API key.
MODEL_NAME = "facebook/bart-large-mnli"

# Score threshold below which we mark the prediction as low-confidence.
# Useful because zero-shot scores can be flat when none of the labels fits
# well - flagging these is more honest than silently picking the top one.
CONFIDENCE_THRESHOLD = 0.40


# -------- Core logic ----------------------------------------------------------

def load_logs(path: Path) -> pd.DataFrame:
    """Load the fault log CSV. Expects a 'log_entry' column."""
    if not path.exists():
        sys.exit(f"Input file not found: {path}")
    df = pd.read_csv(path)
    if "log_entry" not in df.columns:
        sys.exit(f"CSV must contain a 'log_entry' column. Found: {list(df.columns)}")
    return df


def build_classifier():
    """Lazily import transformers so failures here give a clear message."""
    try:
        from transformers import pipeline
    except ImportError:
        sys.exit(
            "transformers is not installed. Run: pip install -r requirements.txt"
        )
    print(f"Loading model '{MODEL_NAME}' (first run downloads ~1.6 GB)...")
    return pipeline("zero-shot-classification", model=MODEL_NAME)


def classify_logs(df: pd.DataFrame, classifier) -> pd.DataFrame:
    """Run each log entry through the classifier and append predictions."""
    predictions = []
    top_scores = []
    n = len(df)
    for i, entry in enumerate(df["log_entry"], start=1):
        result = classifier(entry, candidate_labels=CANDIDATE_LABELS)
        predictions.append(result["labels"][0])
        top_scores.append(round(result["scores"][0], 3))
        print(f"  [{i}/{n}] {result['labels'][0]:<35s} ({result['scores'][0]:.2f})")
    df = df.copy()
    df["predicted_category"] = predictions
    df["confidence"] = top_scores
    df["low_confidence"] = df["confidence"] < CONFIDENCE_THRESHOLD
    return df


def summarise(df: pd.DataFrame) -> str:
    """Build a text summary of the category breakdown."""
    counts = Counter(df["predicted_category"])
    total = len(df)
    low_conf = int(df["low_confidence"].sum())
    lines = ["", "Fault distribution by category", "-" * 40]
    for label in CANDIDATE_LABELS:
        n = counts.get(label, 0)
        pct = (n / total * 100) if total else 0.0
        lines.append(f"  {label:<38s} {n:>3d}  ({pct:5.1f}%)")
    lines.append("-" * 40)
    lines.append(f"  {'Total entries':<38s} {total:>3d}")
    lines.append(f"  {'Low-confidence predictions':<38s} {low_conf:>3d}")
    return "\n".join(lines)


def save_chart(df: pd.DataFrame, path: Path) -> None:
    """Optional bar chart of category counts. matplotlib is an optional dep."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping chart.")
        return
    counts = Counter(df["predicted_category"])
    labels = CANDIDATE_LABELS
    values = [counts.get(label, 0) for label in labels]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, values, color="#3b82a4", edgecolor="#1f2937")
    ax.set_ylabel("Number of log entries")
    ax.set_title("Fault distribution by category")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"Chart saved to {path}")


# -------- CLI -----------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify robot fault log entries by failure type "
                    "using a zero-shot NLI model."
    )
    parser.add_argument(
        "--input", "-i", type=Path, default=Path("logs.csv"),
        help="Input CSV with a 'log_entry' column (default: logs.csv)",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Optional output CSV path for classified results",
    )
    parser.add_argument(
        "--chart", "-c", type=Path, default=None,
        help="Optional path to save a bar chart of the category breakdown",
    )
    args = parser.parse_args()

    df = load_logs(args.input)
    print(f"Loaded {len(df)} log entries from {args.input}")

    classifier = build_classifier()
    print("Classifying entries...")
    df = classify_logs(df, classifier)

    print(summarise(df))

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nResults written to {args.output}")

    if args.chart:
        save_chart(df, args.chart)


if __name__ == "__main__":
    main()
