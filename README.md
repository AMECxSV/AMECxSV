# AMECxSV

Adaptive metadata-driven embedding-fusion calibration for X-lingual speaker
verification.

[Project website](https://amecxsv.github.io/AMECxSV/) |
[Dataset](https://huggingface.co/datasets/AMECxSV/AMECxSV)

All speaker encoders are frozen. AMECxSV studies calibration and fusion of
fixed encoder scores, not representation learning or encoder ranking.

The repository includes single-score baselines, MultiScore-FC/ABS controls,
AMEC-FC/ABS, strict architecture-matched score-only and metadata controls,
linear controls, predicted-language controls, speaker-clustered bootstrap
analysis, and external fixed-score experiments.

Results use a deterministic speaker-disjoint held-out partition derived from
the TidyVoiceX-ASV development protocol. They are metadata-available results,
not language-blind scoring on the official evaluation set.

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
python run.py linear_controls
python experiments/predicted_language_control.py --help
```

The predicted-language control first requires utterance-level language
predictions from a frozen LID model:

```bash
pip install -r requirements-embeddings.txt
python data_prep/predict_voxlingua_languages.py --help
```

External systems are retrained independently on each source calibration split
and support matched score-only and score-plus-metadata inputs:

```bash
python external/train_amec_head_external.py ... --feature-set score_only
python external/train_amec_head_external.py ... --feature-set score_plus_metadata
```
