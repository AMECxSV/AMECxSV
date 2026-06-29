# AMECxSV

Metadata-aware score calibration, fusion, and selective abstention for
cross-lingual automatic speaker verification.

[Project website](https://amecxsv.github.io/AMECxSV/) |
[Dataset](https://huggingface.co/datasets/AMECxSV/AMECxSV)

All speaker encoders are frozen, and absolute score performance remains
encoder-dependent. AMECxSV studies calibration and fusion of fixed encoder
scores; it does not introduce a new encoder or claim the highest raw ASV score.

The repository includes single-score baselines, MultiScore-FC/ABS controls,
AMEC-FC/ABS, strict architecture-matched score-only and metadata controls,
speaker-clustered bootstrap analysis, and external fixed-score experiments.

## Structure

- `experiments/`: internal baselines, AMEC, ablations, controls, and abstention
- `data_prep/`: protocol and score preparation
- `embedding_extraction/`: fixed-encoder wrappers
- `external/`: Official TidyVoice and LI-MSV experiments
- `similarity/`: similarity-fusion experiments

## Setup

```bash
pip install -r requirements.txt
```

## Data

Download the [processed dataset](https://huggingface.co/datasets/AMECxSV/AMECxSV)
into `dataset/`, then set:

```bash
export AMECXSV_INPUT_TABLE=dataset/tidyvoice_trials.parquet
export AMECXSV_FOLLOWUP_TABLE=dataset/tidyvoice_trial_metadata.parquet
```

## Run

```bash
python run.py --list
python run.py --main
python run.py --followup
python run.py matched_score_control
python run.py c9
```

External heads support matched score-only and score-plus-metadata inputs:

```bash
python external/train_amec_head_external.py ... --feature-set score_only
python external/train_amec_head_external.py ... --feature-set score_plus_metadata
```
