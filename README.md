# AMECxSV

Metadata-aware score calibration, fusion, and selective abstention for
cross-lingual automatic speaker verification.

[Project website](https://amecxsv.github.io/AMECxSV/) |
[Dataset](https://huggingface.co/datasets/AMECxSV/AMECxSV)

All speaker encoders are frozen, and absolute score performance remains
encoder-dependent. AMECxSV studies calibration and fusion of fixed encoder
scores; it does not introduce a new encoder or claim the highest raw ASV score.

C0--C5 are single-encoder baselines covering raw scoring, classical calibration,
and condition-aware calibration. C8 is the score-only fusion control, and C10
adds metadata as calibration context.

## Structure

- `experiments/`: C0--C5, C8, C9, C10, ablations, and controls
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
python run.py c9
```
