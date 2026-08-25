"""Compare only transformer input-window length on the no-COVID CES setup."""

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


WINDOW_LENGTHS = (6, 12, 24)
COVID_EXCLUDE_START = "2020-03"
COVID_EXCLUDE_END = "2021-12"

SUMMARY_PATH = baseline.ROOT / "output" / "train_ces_state_window_compare_summary.txt"
RESULTS_PATH = baseline.ROOT / "output" / "train_ces_state_window_compare.csv"


def build_windows_for_length(
    data: pd.DataFrame, window_length: int, target_start: str, target_end: str | None = None
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build scaled one-step windows for one explicit input length."""
    if window_length <= 0:
        raise ValueError("window_length must be positive")
    start = pd.Period(target_start, freq="M").to_timestamp()
    end = pd.Period(target_end, freq="M").to_timestamp() if target_end else None
    inputs: list[np.ndarray] = []
    targets: list[float] = []
    rows: list[dict[str, object]] = []
    for _, group in data.groupby("state_fips", sort=True):
        group = group.sort_values("date", kind="mergesort").reset_index(drop=True)
        values = group["scaled_log_difference"].to_numpy(dtype=float)
        for index in range(window_length, len(group)):
            target_date = group.loc[index, "date"]
            if target_date < start or (end is not None and target_date > end):
                continue
            window = values[index - window_length : index]
            target = values[index]
            if not np.isfinite(window).all() or not np.isfinite(target):
                continue
            inputs.append(window[:, None])
            targets.append(float(target))
            rows.append(
                {
                    "state_fips": group.loc[index, "state_fips"],
                    "state_name": group.loc[index, "state_name"],
                    "forecast_date": target_date,
                    "previous_level": float(group.loc[index - 1, "level"]),
                    "actual_level": float(group.loc[index, "level"]),
                    "actual_log_diff": float(group.loc[index, "log_difference"]),
                }
            )
    if not inputs:
        raise ValueError("no usable windows were constructed")
    return np.asarray(inputs, dtype=np.float32), np.asarray(targets, dtype=np.float32)[:, None], pd.DataFrame(rows)


def filter_excluded_targets(X: np.ndarray, y: np.ndarray, metadata: pd.DataFrame):
    """Apply the no-COVID target-date rule without deleting source rows."""
    dates = metadata["forecast_date"]
    start = pd.Period(COVID_EXCLUDE_START, freq="M").to_timestamp()
    end = pd.Period(COVID_EXCLUDE_END, freq="M").to_timestamp()
    keep = ~dates.between(start, end)
    return X[keep.to_numpy()], y[keep.to_numpy()], metadata.loc[keep].reset_index(drop=True)


def _state_metrics(metadata: pd.DataFrame, actual: np.ndarray, predictions: np.ndarray) -> pd.DataFrame:
    rows = []
    for state_name, group in metadata.groupby("state_name", sort=True):
        indices = group.index.to_numpy()
        error = predictions[indices] - actual[indices]
        rows.append({"state_name": state_name, "rmse": float(np.sqrt(np.mean(error**2))), "mae": float(np.mean(np.abs(error)))})
    return pd.DataFrame(rows)


def _metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return baseline._metrics(actual, prediction)


def _train_one(X_train: np.ndarray, y_train: np.ndarray, X_holdout: np.ndarray) -> tuple[np.ndarray, list[float], int, str]:
    """Train one fresh model with the unchanged baseline settings."""
    np.random.seed(baseline.SEED)
    torch.manual_seed(baseline.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(baseline.SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    loader_generator = torch.Generator().manual_seed(baseline.SEED)
    loader = DataLoader(dataset, batch_size=baseline.BATCH_SIZE, shuffle=True, generator=loader_generator)
    model = baseline.TimeSeriesTransformer(baseline.D_MODEL, baseline.D_FF, baseline.N_BLOCKS).to(device)
    loss_fn = baseline.make_loss(baseline.LOSS)
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
            print(f"epoch {epoch + 1:3d}/{baseline.EPOCHS}: loss={history[-1]:.8f}")
    model.eval()
    with torch.inference_mode():
        prediction = model(torch.from_numpy(X_holdout).to(device)).cpu().numpy().reshape(-1)
    return prediction, history, sum(parameter.numel() for parameter in model.parameters()), str(device)


def _tail(values: np.ndarray) -> dict[str, float | int]:
    probabilities = [0, 0.01, 0.5, 0.99, 1]
    quantiles = np.quantile(values, probabilities)
    return {
        "min_prediction": float(quantiles[0]),
        "p01_prediction": float(quantiles[1]),
        "median_prediction": float(quantiles[2]),
        "p99_prediction": float(quantiles[3]),
        "max_prediction": float(quantiles[4]),
        "count_below_-0.01": int((values < -0.01).sum()),
        "count_below_-0.02": int((values < -0.02).sum()),
        "count_below_-0.05": int((values < -0.05).sum()),
    }


def main() -> None:
    data = baseline.load_and_validate_ces()
    train_rows, holdout_rows = baseline.chronological_split(data)
    eligible_rows = eligible_training_rows(data)
    scaling = baseline.fit_state_scaling(eligible_rows)
    scaled = baseline.apply_state_scaling(data, scaling)
    coefficients = fit_ar1_excluding(data)
    ar1_ordered = baseline.forecast_ar1(data, coefficients, baseline.HOLDOUT_START)
    rw_growth = np.zeros(len(holdout_rows))

    baseline_diag = pd.read_csv(baseline.DIAGNOSTICS_PATH, parse_dates=["forecast_date"])
    results: list[dict[str, object]] = []
    details: list[tuple[int, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, list[float], int, str]] = []
    for window_length in WINDOW_LENGTHS:
        X_all, y_all, all_metadata = build_windows_for_length(scaled, window_length, "2005-01", baseline.TRAIN_END)
        X_train, y_train, train_metadata = filter_excluded_targets(X_all, y_all, all_metadata)
        X_holdout, _, metadata = build_windows_for_length(scaled, window_length, baseline.HOLDOUT_START)
        if len(metadata) != len(holdout_rows):
            raise RuntimeError("window length changed the holdout row count")
        prediction_scaled, history, parameter_count, device = _train_one(X_train, y_train, X_holdout)
        scaling_by_row = metadata[["state_fips"]].merge(scaling, on="state_fips", how="left", sort=False, validate="many_to_one")
        prediction_growth = prediction_scaled * scaling_by_row["std"].to_numpy() + scaling_by_row["mean"].to_numpy()
        actual_growth = metadata["actual_log_diff"].to_numpy()
        previous_level = metadata["previous_level"].to_numpy()
        actual_level = metadata["actual_level"].to_numpy()
        ar1_growth = ar1_ordered
        rw = rw_growth
        transformer_level = baseline.invert_log_difference(prediction_growth, previous_level)
        ar1_level = baseline.invert_log_difference(ar1_growth, previous_level)
        rw_level = previous_level.copy()
        state = _state_metrics(metadata, actual_level, transformer_level)
        results.append(
            {
                "window": window_length,
                "training_windows": len(X_train),
                "growth_rmse": _metrics(actual_growth, prediction_growth)["rmse"],
                "growth_mae": _metrics(actual_growth, prediction_growth)["mae"],
                "level_rmse": _metrics(actual_level, transformer_level)["rmse"],
                "level_mae": _metrics(actual_level, transformer_level)["mae"],
                "median_state_rmse": state["rmse"].median(),
                "states_beating_ar1": int((state["rmse"] < _state_metrics(metadata, actual_level, ar1_level)["rmse"]).sum()),
                "states_beating_rw": int((state["rmse"] < _state_metrics(metadata, actual_level, rw_level)["rmse"]).sum()),
                **_tail(prediction_growth),
                "initial_loss": history[0],
                "final_loss": history[-1],
                "minimum_loss": min(history),
            }
        )
        details.append((window_length, metadata, actual_level, transformer_level, prediction_growth, history, parameter_count, device))
        del X_all, y_all, X_train, y_train, X_holdout
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results_df = pd.DataFrame(results).sort_values("window").reset_index(drop=True)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_PATH, index=False)
    baseline_growth = {name: _metrics(baseline_diag.actual_log_diff, baseline_diag[f"{name}_log_diff"]) for name in ["transformer", "ar1", "rw"]}
    baseline_level = {name: _metrics(baseline_diag.actual_level, baseline_diag[f"{name}_level"]) for name in ["transformer", "ar1", "rw"]}
    ar1_growth_metrics = _metrics(details[0][2] * 0 + pd.read_csv(baseline.DIAGNOSTICS_PATH).actual_log_diff.to_numpy(), ar1_ordered)
    year_lines = []
    for year in [2023, 2024, 2025, 2026]:
        line = {"year": year}
        for window_length, metadata, actual_level, transformer_level, _, _, _, _ in details:
            mask = metadata.forecast_date.dt.year.to_numpy() == year
            line[f"window_{window_length}_rmse"] = _metrics(actual_level[mask], transformer_level[mask])["rmse"]
        year_lines.append(line)

    summary = [
        "CES State total-nonfarm transformer window-length comparison",
        f"windows: {WINDOW_LENGTHS}",
        f"covid_exclusion: {COVID_EXCLUDE_START} through {COVID_EXCLUDE_END}",
        f"train_end: {baseline.TRAIN_END}",
        f"holdout: {baseline.HOLDOUT_START} through {holdout_rows.date.max().strftime('%Y-%m')}",
        f"constants: d_model={baseline.D_MODEL}, d_ff={baseline.D_FF}, n_blocks={baseline.N_BLOCKS}, batch_size={baseline.BATCH_SIZE}, learning_rate={baseline.LEARNING_RATE}, epochs={baseline.EPOCHS}, seed={baseline.SEED}, loss={baseline.LOSS}",
        f"device: {details[0][7]}",
        f"transformer_parameters: {details[0][6]}",
        "benchmark_growth_metrics:",
        *(f"  {name}: {values}" for name, values in {**baseline_growth, "no_covid_ar1_recomputed": ar1_growth_metrics}.items()),
        "benchmark_level_metrics:",
        *(f"  {name}: {values}" for name, values in baseline_level.items()),
        "results:",
        *[f"  {row.to_dict()}" for _, row in results_df.iterrows()],
        "yearly_level_rmse:",
        *[f"  {line}" for line in year_lines],
        "rankings:",
        f"  growth_rmse: {results_df.sort_values('growth_rmse').window.tolist()}",
        f"  level_rmse: {results_df.sort_values('level_rmse').window.tolist()}",
        f"  median_state_rmse: {results_df.sort_values('median_state_rmse').window.tolist()}",
        "window_attention_cost_relative_to_6: 6=1x, 12=4x, 24=16x",
        "baseline_no_covid_12_month_reference:",
        f"  growth_rmse={baseline_growth['transformer']['rmse']:.10f}, level_rmse={baseline_level['transformer']['rmse']:.10f}",
    ]
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(results_df.to_string(index=False))
    print("yearly level RMSE:")
    print(pd.DataFrame(year_lines).to_string(index=False))
    print(f"summary: {SUMMARY_PATH}")
    print(f"results: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
