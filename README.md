# Robot Fault Log Classifier

A small Python CLI tool that categorises robot fault log entries into
failure types using a zero-shot natural-language inference model from
Hugging Face. No fine-tuning, no API key, no GPU required.

## Motivation

This project grew out of manual fault analysis work I did during a
year-long group project on a coffee-making robot demonstrator at the
University of Sheffield's Advanced Manufacturing Research Centre
(AMRC), in which an IGUS ReBeL 6-DOF arm was driven by ROS2 and the
MoveIt Task Constructor framework.

During development I spent a lot of time reading raw runtime logs by
hand — diagnosing planner failures, inverse-kinematics rejections,
collision events and joint-limit violations. Once the project was
finished I wanted a small, honest exercise in applied AI to automate
the first-pass categorisation. The result is this tool: a single
script, ~30 hand-written log entries based on real issues I
encountered, and a zero-shot classifier that sorts them into four
non-overlapping fault categories.

## What it does

- Loads a CSV of fault log entries (`logs.csv`)
- Runs each entry through the
  [`facebook/bart-large-mnli`](https://huggingface.co/facebook/bart-large-mnli)
  zero-shot classification pipeline
- Predicts one of four failure categories and reports the model's
  confidence
- Flags low-confidence predictions (below 0.40) for manual review
- Prints a summary breakdown to the console
- Optionally writes results to a CSV and saves a bar chart

The four categories are:

| Category | Typical log content |
| --- | --- |
| `motion planning failure` | RRT-Connect / Cartesian planner could not find a path |
| `collision detection event` | A collision was detected in the planning scene |
| `inverse kinematics failure` | The IK solver returned no valid solution |
| `joint or velocity limit violation` | A joint, velocity, acceleration or torque limit was exceeded |

## How it works

[BART-MNLI](https://huggingface.co/facebook/bart-large-mnli) is a
pre-trained natural-language inference model. Zero-shot classification
uses it by treating each candidate label as a hypothesis
("*This text is about motion planning failure*") and scoring how well
the input log entry entails each hypothesis. The label with the
highest entailment score is returned as the prediction.

This approach is appropriate here for two reasons:

1. With only ~30 log entries there isn't enough data to train a
   custom classifier.
2. The fault categories are described in natural language that the
   pre-trained model already understands, so the model can score them
   without ever having seen robot logs in training.

The trade-off is that zero-shot scores depend on the *wording* of the
labels. The labels in `classify.py` were chosen to be short,
descriptive and non-overlapping for that reason.

## Setup

```bash
git clone https://github.com/Tkayy0/fault-log-classifier.git
cd fault-log-classifier
pip install -r requirements.txt
```

The first run downloads the model (~1.6 GB) from the Hugging Face
hub and caches it locally. After that it runs offline.

## Usage

```bash
# Classify the included sample logs and print the summary
python classify.py --input logs.csv

# Save results to a CSV
python classify.py --input logs.csv --output results.csv

# Also produce a bar chart of the distribution
python classify.py --input logs.csv --output results.csv --chart fault_breakdown.png
```

## Example output

Running on the 30 included log entries produces output like the
following (full transcript in
[`examples/sample_console_output.txt`](examples/sample_console_output.txt)):

```
Loaded 30 log entries from logs.csv
Loading model 'facebook/bart-large-mnli' (first run downloads ~1.6 GB)...
Classifying entries...
  [1/30] motion planning failure              (0.69)
  [2/30] collision detection event            (0.82)
  [3/30] inverse kinematics failure           (0.78)
  ...

Fault distribution by category
----------------------------------------
  motion planning failure                  7  ( 23.3%)
  collision detection event               10  ( 33.3%)
  inverse kinematics failure               6  ( 20.0%)
  joint or velocity limit violation        7  ( 23.3%)
----------------------------------------
  Total entries                           30
  Low-confidence predictions               0
```

![Fault breakdown chart]<img width="1200" height="675" alt="fault_breakdown" src="https://github.com/user-attachments/assets/d343bf2e-e9ac-4979-ab01-a74d7f91f1ac" />


## Repository layout

```
fault-log-classifier/
├── README.md
├── requirements.txt
├── logs.csv                # 30 hand-written log entries
├── classify.py             # the CLI script
└── examples/
    ├── sample_console_output.txt
    ├── sample_classified_results.csv
    └── fault_breakdown.png
```

## Honest limitations

A few things worth being upfront about:

- **The log entries are hand-written.** They are based on real issues
  I encountered during the AMRC project, but they are not raw exports
  from a logging system. A production version would need to handle
  noisier real-world text.
- **No quantitative accuracy claim is made.** With 30 examples and no
  separate ground-truth labelling pass, reporting an accuracy figure
  would not be meaningful. The tool is useful as a first-pass
  categoriser; a human should still review the results.
- **Zero-shot classification is not magic.** When none of the
  candidate labels fits the input well, the model will still pick
  one — that is why the `confidence` column and `low_confidence`
  flag exist.

## Future improvements

- Add a labelled evaluation set and report accuracy / per-class
  precision and recall
- Support log inputs from JSON or directly from ROS2 bag files
- Try a smaller, faster model (e.g. `valhalla/distilbart-mnli-12-3`)
  to reduce the download size and inference time
- Add a sub-category prediction step for the most common category

## Author

Built by Keegan Chinyenze as a follow-up to the ELE322 group project
at the University of Sheffield.

## Licence

MIT
