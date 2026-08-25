"""Compare transformer width on the fixed no-COVID CES experiment."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_ces_state_holdout as baseline
from train_ces_state_holdout_no_covid import eligible_training_rows, fit_ar1_excluding
from train_ces_state_window_compare import build_windows_for_length, filter_excluded_targets


WINDOW_LENGTH = 6
D_MODEL_VALUES = (4, 8, 16)
N_BLOCKS = 2
LOSS = "huber"
COVID_EXCLUDE_START = "2020-03"
COVID_EXCLUDE_END = "2021-12"

SUMMARY_PATH = baseline.ROOT / "output" / "train_ces_state_size_compare_summary.txt"
RESULTS_PATH = baseline.ROOT / "output" / "train_ces_state_size_compare.csv"
STATE_RESULTS_PATH = baseline.ROOT / "output" / "train_ces_state_size_compare_by_state.csv"


def model_config(d_model: int) -> tuple[int, int]:
    """Return the requested width and coupled feed-forward dimension."""
    if d_model <= 0:
        raise ValueError("d_model must be positive")
    return d_model, 2 * d_model


def _metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return baseline._metrics(actual, prediction)


def _tail(values: np.ndarray) -> dict[str, float | int]:
    quantiles = np.quantile(values, [0, 0.01, 0.5, 0.99, 1])
    return {
        "min_prediction": float(quantiles[0]),
        "p01_prediction": float(quantiles[1]),
        "median_prediction": float(quantiles[2]),
        "p99_prediction": float(quantiles[3]),
        "max_prediction": float(quantiles[4]),
        "count_below_-0.01": int((values < -0.01).sum()),
        "count_above_0.01": int((values > 0.01).sum()),
    }


def _fit_one(X_train: np.ndarray, y_train: np.ndarray, X_holdout: np.ndarray, d_model: int):
    d_model, d_ff = model_config(d_model)
    np.random.seed(baseline.SEED)
    torch.manual_seed(baseline.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(baseline.SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    loader_generator = torch.Generator().manual_seed(baseline.SEED)
    loader = DataLoader(dataset, batch_size=baseline.BATCH_SIZE, shuffle=True, generator=loader_generator)
    model = baseline.TimeSeriesTransformer(d_model, d_ff, N_BLOCKS).to(device)
    loss_fn = baseline.make_loss(LOSS)
    optimizer = torch.optim.AdamW(model.parameters(), lr=baseline.LEARNING_RATE)
    history: list[float] = []
    for epoch in range(baseline.EPOCHS):
        model.train()
        total_loss = 0.0
        total_items = 0
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
            count = len(batch_X)
            total_loss += loss.item() * count
            total_items += count
        history.append(total_loss / total_items)
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == baseline.EPOCHS - 1:
            print(f"d_model={d_model} epoch {epoch + 1:3d}/{baseline.EPOCHS}: loss={history[-1]:.8f}")
    model.eval()
    with torch.inference_mode():
        prediction = model(torch.from_numpy(X_holdout).to(device)).cpu().numpy().reshape(-1)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    return prediction, history, parameter_count, str(device)


def _state_table(metadata: pd.DataFrame, actual: np.ndarray, levels: dict[int, np.ndarray], ar1: np.ndarray, rw: np.ndarray) -> pd.DataFrame:
    rows = []
    for state_name, group in metadata.groupby("state_name", sort=True):
        indices = group.index.to_numpy()
        actual_state = actual[indices]
        row = {"state_name": state_name}
        for d_model, values in levels.items():
            row[f"d{d_model}_rmse"] = _metrics(actual_state, values[indices])["rmse"]
            row[f"d{d_model}_mae"] = _metrics(actual_state, values[indices])["mae"]
        row["ar1_rmse"] = _metrics(actual_state, ar1[indices])["rmse"]
        row["rw_rmse"] = _metrics(actual_state, rw[indices])["rmse"]
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    data = baseline.load_and_validate_ces()
    train_rows, holdout_rows = baseline.chronological_split(data)
    eligible_rows = eligible_training_rows(data)
    scaling = baseline.fit_state_scaling(eligible_rows)
    scaled = baseline.apply_state_scaling(data, scaling)
    X_all, y_all, all_metadata = build_windows_for_length(scaled, WINDOW_LENGTH, "2005-01", baseline.TRAIN_END)
    X_train, y_train, _ = filter_excluded_targets(X_all, y_all, all_metadata)
    X_holdout, _, metadata = build_windows_for_length(scaled, WINDOW_LENGTH, baseline.HOLDOUT_START)
    if len(X_train) != 9537 or len(X_holdout) != 2193:
        raise RuntimeError("unexpected training or holdout window count")

    coefficients = fit_ar1_excluding(data)
    ar1_growth = baseline.forecast_ar1(data, coefficients, baseline.HOLDOUT_START)
    actual_growth = metadata["actual_log_diff"].to_numpy()
    previous_level = metadata["previous_level"].to_numpy()
    actual_level = metadata["actual_level"].to_numpy()
    ar1_level = baseline.invert_log_difference(ar1_growth, previous_level)
    rw_level = previous_level.copy()
    predictions: dict[int, np.ndarray] = {}
    levels: dict[int, np.ndarray] = {}
    histories: dict[int, list[float]] = {}
    parameter_counts: dict[int, int] = {}
    device = ""
    for d_model in D_MODEL_VALUES:
        prediction_scaled, history, parameter_count, device = _fit_one(X_train, y_train, X_holdout, d_model)
        row_scaling = metadata[["state_fips"]].merge(scaling, on="state_fips", how="left", sort=False, validate="many_to_one")
        predictions[d_model] = prediction_scaled * row_scaling["std"].to_numpy() + row_scaling["mean"].to_numpy()
        levels[d_model] = baseline.invert_log_difference(predictions[d_model], previous_level)
        histories[d_model] = history
        parameter_counts[d_model] = parameter_count

    state_table = _state_table(metadata, actual_level, levels, ar1_level, rw_level)
    state_table.to_csv(STATE_RESULTS_PATH, index=False)
    reference_parameters = parameter_counts[8]
    rows = []
    for d_model in D_MODEL_VALUES:
        growth_metrics = _metrics(actual_growth, predictions[d_model])
        level_metrics = _metrics(actual_level, levels[d_model])
        state_rmse = state_table[f"d{d_model}_rmse"]
        rows.append(
            {
                "d_model": d_model,
                "d_ff": 2 * d_model,
                "parameters": parameter_counts[d_model],
                "relative_parameters": parameter_counts[d_model] / reference_parameters,
                "training_windows": len(X_train),
                "growth_rmse": growth_metrics["rmse"],
                "growth_mae": growth_metrics["mae"],
                "level_rmse": level_metrics["rmse"],
                "level_mae": level_metrics["mae"],
                "median_state_rmse": state_rmse.median(),
                "wins_ar1": int((state_rmse < state_table.ar1_rmse).sum()),
                "wins_rw": int((state_rmse < state_table.rw_rmse).sum()),
                **_tail(predictions[d_model]),
                "initial_loss": histories[d_model][0],
                "final_loss": histories[d_model][-1],
                "minimum_loss": min(histories[d_model]),
            }
        )
    results = pd.DataFrame(rows).sort_values("d_model").reset_index(drop=True)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_PATH, index=False)

    yearly = []
    for year in [2023, 2024, 2025, 2026]:
        mask = metadata["forecast_date"].dt.year.to_numpy() == year
        row = {"year": year}
        for d_model in D_MODEL_VALUES:
            row[f"d{d_model}_rmse"] = _metrics(actual_level[mask], levels[d_model][mask])["rmse"]
        yearly.append(row)
    yearly_df = pd.DataFrame(yearly)
    baseline_diag = pd.read_csv(baseline.DIAGNOSTICS_PATH)
    benchmark_growth = {name: _metrics(baseline_diag.actual_log_diff, baseline_diag[f"{name}_log_diff"]) for name in ["ar1", "rw"]}
    benchmark_level = {name: _metrics(baseline_diag.actual_level, baseline_diag[f"{name}_level"]) for name in ["ar1", "rw"]}
    summary = [
        "CES State total-nonfarm model-size comparison",
        f"window_length: {WINDOW_LENGTH}",
        f"loss: {LOSS}",
        f"covid_exclusion: {COVID_EXCLUDE_START} through {COVID_EXCLUDE_END}",
        f"train_end: {baseline.TRAIN_END}",
        f"holdout: {baseline.HOLDOUT_START} through {holdout_rows.date.max().strftime('%Y-%m')}",
        f"constants: n_blocks={N_BLOCKS}, batch_size={baseline.BATCH_SIZE}, learning_rate={baseline.LEARNING_RATE}, epochs={baseline.EPOCHS}, seed={baseline.SEED}",
        f"device: {device}",
        f"training_windows: {len(X_train)}",
        f"parameter_counts: {parameter_counts}",
        f"relative_to_d8: { {d: parameter_counts[d] / reference_parameters for d in D_MODEL_VALUES} }",
        "benchmark_growth_metrics:",
        *(f"  {name}: {value}" for name, value in benchmark_growth.items()),
        "benchmark_level_metrics:",
        *(f"  {name}: {value}" for name, value in benchmark_level.items()),
        "results:",
        *[f"  {row.to_dict()}" for _, row in results.iterrows()],
        "yearly_level_rmse:",
        *[f"  {row.to_dict()}" for _, row in yearly_df.iterrows()],
        "rankings:",
        f"  growth_rmse: {results.sort_values('growth_rmse').d_model.tolist()}",
        f"  level_rmse: {results.sort_values('level_rmse').d_model.tolist()}",
        f"  median_state_rmse: {results.sort_values('median_state_rmse').d_model.tolist()}",
        f"d4_beats_d8: {state_table.loc[state_table.d4_rmse < state_table.d8_rmse, 'state_name'].tolist()}",
        f"d8_beats_d4: {state_table.loc[state_table.d8_rmse < state_table.d4_rmse, 'state_name'].tolist()}",
        f"d16_beats_d8: {state_table.loc[state_table.d16_rmse < state_table.d8_rmse, 'state_name'].tolist()}",
        "training_losses:",
        *[f"  d{d}: initial={histories[d][0]:.10f}, final={histories[d][-1]:.10f}, minimum={min(histories[d]):.10f}" for d in D_MODEL_VALUES],
    ]
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(results.to_string(index=False))
    print("yearly level RMSE:")
    print(yearly_df.to_string(index=False))
    print(f"summary: {SUMMARY_PATH}")
    print(f"results: {RESULTS_PATH}")
    print(f"state results: {STATE_RESULTS_PATH}")


if __name__ == "__main__":
    main()
