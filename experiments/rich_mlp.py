from __future__ import annotations

import math

import numpy as np
from tqdm import tqdm

try:
    import torch
except ImportError:
    torch = None


def validation_cllr(scores: np.ndarray, labels: np.ndarray) -> float:
    labels_bool = labels.astype(bool)
    target_scores = scores[labels_bool]
    nontarget_scores = scores[~labels_bool]
    if target_scores.size == 0 or nontarget_scores.size == 0:
        return math.inf
    target_loss = np.mean(np.logaddexp(0.0, -target_scores))
    nontarget_loss = np.mean(np.logaddexp(0.0, nontarget_scores))
    return float(
        0.5 * (target_loss + nontarget_loss) / math.log(2.0)
    )


class RichFeatureMlpCalibrator:
    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int,
        dropout: float,
        learning_rate: float,
        c_value: float,
        epochs: int,
        batch_size: int,
        device,
        num_scores: int = 6,
        name: str | None = None,
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required")
        if input_dim < num_scores:
            raise ValueError(
                "Expected score columns followed by optional metadata columns"
            )
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.c_value = c_value
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.num_scores = num_scores
        self.name = name or (
            f"rich_feature_mlp_{input_dim}d_h{hidden_dim}_drop{dropout:g}"
        )
        self.seed = 0
        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None
        self.net = None

    def augment(self, features: np.ndarray) -> np.ndarray:
        scores = features[:, : self.num_scores]
        meta = features[:, self.num_scores :]
        score_mean = np.mean(scores, axis=1, keepdims=True)
        score_std = np.std(scores, axis=1, keepdims=True)
        score_min = np.min(scores, axis=1, keepdims=True)
        score_max = np.max(scores, axis=1, keepdims=True)
        score_range = score_max - score_min
        sorted_scores = np.sort(scores, axis=1)[:, ::-1]
        top_gap = (
            sorted_scores[:, :1] - sorted_scores[:, 1:2]
            if self.num_scores > 1
            else np.zeros_like(score_mean)
        )
        centered_scores = scores - score_mean
        interactions = (
            scores[:, :, None] * meta[:, None, :]
        ).reshape(features.shape[0], -1)

        adjacent_gaps = sorted_scores[:, :-1] - sorted_scores[:, 1:]

        def linear_quantile_from_descending(q: float) -> np.ndarray:
            position = (self.num_scores - 1) * q
            lower = int(math.floor(position))
            upper = int(math.ceil(position))
            weight = position - lower
            lower_desc = self.num_scores - 1 - lower
            upper_desc = self.num_scores - 1 - upper
            lower_value = sorted_scores[:, lower_desc : lower_desc + 1]
            if lower == upper:
                return lower_value
            upper_value = sorted_scores[:, upper_desc : upper_desc + 1]
            difference = upper_value - lower_value
            if weight >= 0.5:
                return upper_value - difference * (1.0 - weight)
            return lower_value + difference * weight

        middle = self.num_scores // 2
        if self.num_scores % 2:
            score_median = sorted_scores[:, middle : middle + 1]
        else:
            score_median = np.mean(
                sorted_scores[:, middle - 1 : middle + 1],
                axis=1,
                keepdims=True,
            )
        score_q25 = linear_quantile_from_descending(0.25)
        score_q75 = linear_quantile_from_descending(0.75)
        pairwise_diffs = []
        pairwise_abs_diffs = []
        pairwise_products = []
        for left in range(self.num_scores):
            for right in range(left + 1, self.num_scores):
                diff = (
                    scores[:, left : left + 1]
                    - scores[:, right : right + 1]
                )
                pairwise_diffs.append(diff)
                pairwise_abs_diffs.append(np.abs(diff))
                pairwise_products.append(
                    scores[:, left : left + 1]
                    * scores[:, right : right + 1]
                )

        return np.concatenate(
            [
                features,
                score_mean,
                score_std,
                score_min,
                score_max,
                score_range,
                sorted_scores[:, :3],
                top_gap,
                centered_scores,
                interactions,
                sorted_scores,
                adjacent_gaps,
                score_median,
                score_q25,
                score_q75,
                *pairwise_diffs,
                *pairwise_abs_diffs,
                *pairwise_products,
                meta * meta,
            ],
            axis=1,
        )

    def _build_net(self, input_dim: int):
        return torch.nn.Sequential(
            torch.nn.Linear(input_dim, self.hidden_dim),
            torch.nn.LayerNorm(self.hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.LayerNorm(self.hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(self.hidden_dim, 1),
        )

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        sample_weights: np.ndarray,
        *,
        validation_features: np.ndarray | None = None,
        validation_labels: np.ndarray | None = None,
        early_stopping_patience: int = 60,
        scheduler_patience: int = 10,
        scheduler_factor: float = 0.5,
        min_learning_rate: float = 1.0e-5,
    ) -> None:
        torch.manual_seed(self.seed)
        features = self.augment(features)
        validation_features = (
            None
            if validation_features is None
            else self.augment(validation_features)
        )
        x_tensor = torch.as_tensor(
            features, dtype=torch.float32, device=self.device
        )
        y_tensor = torch.as_tensor(
            labels.astype(np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        w_tensor = torch.as_tensor(
            sample_weights.astype(np.float32),
            dtype=torch.float32,
            device=self.device,
        )

        self.mean = x_tensor.mean(dim=0, keepdim=True)
        self.std = x_tensor.std(
            dim=0, keepdim=True, unbiased=False
        ).clamp_min(1.0e-6)
        x_scaled = (x_tensor - self.mean) / self.std
        validation_scaled = None
        if validation_features is not None:
            validation_tensor = torch.as_tensor(
                validation_features,
                dtype=torch.float32,
                device=self.device,
            )
            validation_scaled = (
                validation_tensor - self.mean
            ) / self.std

        self.net = self._build_net(x_scaled.shape[1]).to(self.device)
        optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=self.learning_rate,
            weight_decay=1.0 / max(self.c_value, 1.0e-12),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=scheduler_factor,
            patience=scheduler_patience,
            threshold=1.0e-5,
            min_lr=min_learning_rate,
        )

        n_rows = int(x_scaled.shape[0])
        batch_size = min(self.batch_size, n_rows)
        best_cllr = math.inf
        best_epoch = -1
        best_state = None
        epochs_without_improvement = 0

        for epoch in tqdm(
            range(self.epochs), desc=f"train {self.name}", unit="epoch"
        ):
            order = torch.randperm(n_rows, device=self.device)
            self.net.train()
            for start in range(0, n_rows, batch_size):
                idx = order[start : start + batch_size]
                logits = self.net(x_scaled[idx]).squeeze(1)
                loss_terms = (
                    torch.nn.functional.binary_cross_entropy_with_logits(
                        logits, y_tensor[idx], reduction="none"
                    )
                )
                weights = w_tensor[idx]
                loss = torch.sum(weights * loss_terms) / torch.sum(
                    weights
                ).clamp_min(1.0e-12)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            if validation_scaled is None or validation_labels is None:
                continue
            self.net.eval()
            outputs = []
            with torch.no_grad():
                for start in range(0, validation_scaled.shape[0], 1_000_000):
                    logits = self.net(
                        validation_scaled[start : start + 1_000_000]
                    ).squeeze(1)
                    outputs.append(
                        logits.detach()
                        .cpu()
                        .numpy()
                        .astype(np.float64, copy=False)
                    )
            current_cllr = validation_cllr(
                np.concatenate(outputs), validation_labels
            )
            scheduler.step(current_cllr)
            if current_cllr < best_cllr - 1.0e-5:
                best_cllr = current_cllr
                best_epoch = epoch + 1
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.net.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= early_stopping_patience:
                    break

        self.net.eval()
        if best_state is not None:
            self.net.load_state_dict(best_state)
            self.net.eval()
            print(
                f"{self.name} restored best_epoch={best_epoch} "
                f"best_val_Cllr={best_cllr:.6f}",
                flush=True,
            )

    def decision_function(
        self, features: np.ndarray, batch_size: int = 1_000_000
    ) -> np.ndarray:
        if self.mean is None or self.std is None or self.net is None:
            raise RuntimeError("Model is not fitted")
        features = self.augment(features)
        outputs = []
        with torch.no_grad():
            for start in range(0, features.shape[0], batch_size):
                batch = torch.as_tensor(
                    features[start : start + batch_size],
                    dtype=torch.float32,
                    device=self.device,
                )
                batch = (batch - self.mean) / self.std
                logits = self.net(batch).squeeze(1)
                outputs.append(
                    logits.detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64, copy=False)
                )
        return np.concatenate(outputs)


rich_feature_mlp = RichFeatureMlpCalibrator
