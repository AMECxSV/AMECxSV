# AMECxSV

Anonymous implementation of metadata-aware calibration and selective
abstention for cross-lingual automatic speaker verification.

## Structure

- `experiments/`: C0--C5, C8, C9, C10, ablations, and controls
- `data_prep/`: protocol, split, embedding, and score preparation
- `embedding_extraction/`: wrappers for six fixed speaker encoders
- `external/`: external baseline evaluation
- `similarity/`: similarity-fusion experiments
- `pages/`: website figures

## Setup

```bash
pip install -r requirements.txt
```

Install `requirements-embeddings.txt` only when extracting embeddings.

## Data

Place prepared files under `dataset/`. Data, pretrained checkpoints, and
generated outputs are not included.

Set the main input table when needed:

```bash
export AMECXSV_INPUT_TABLE=dataset/tidyvoice_trials.parquet
```

## Run

```bash
python run.py --list
python run.py --main
python run.py --followup
python run.py c9
```
