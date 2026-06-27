from __future__ import annotations

from common import BASELINE_DIR, BEST_MLP_SETTING, EMBEDDINGS, Experiment, run_experiments


OUTPUT_CSV = BASELINE_DIR / "tidyvoice_c3_results.csv"
CHUNKSIZE = 250_000
MAX_TRAIN_ROWS_PER_CLASS = None
MAX_EVAL_ROWS_PER_CLASS = None
DECISION_PRIOR = 0.01
MODEL_SETTING = {key: value for key, value in BEST_MLP_SETTING.items() if key != "c_value"}


def main() -> None:
    experiments = [Experiment("C3", "c3_qmf", BEST_MLP_SETTING["c_value"], **MODEL_SETTING)]
    run_experiments(
        output_csv=OUTPUT_CSV,
        embeddings=EMBEDDINGS,
        experiments=experiments,
        chunksize=CHUNKSIZE,
        max_train_rows_per_class=MAX_TRAIN_ROWS_PER_CLASS,
        max_eval_rows_per_class=MAX_EVAL_ROWS_PER_CLASS,
        decision_prior=DECISION_PRIOR,
    )


if __name__ == "__main__":
    main()
