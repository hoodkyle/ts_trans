"""Compare MSE and default Huber loss on the fixed no-COVID CES setup."""

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
LOSSES = ("mse", "huber")
COVID_EXCLUDE_START = "2020-03"
COVID_EXCLUDE_END = "2021-12"

SUMMARY_PATH = baseline.ROOT / "output" / "train_ces_state_loss_compare_summary.txt"
RESULTS_PATH = baseline.ROOT / "output" / "train_ces_state_loss_compare.csv"
STATE_RESULTS_PATH = baseline.ROOT / "output" / "train_ces_state_loss_compare_by_state.csv"


def _metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return baseline._metrics(actual, prediction)


def _tail(values: np.ndarray) -> dict[str, float | int]:
    probabilities = [0, 0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999, 1]
    quantiles = np.quantile(values, probabilities)
    return {
        "min_prediction": float(quantiles[0]),
        "p01pct_prediction": float(quantiles[1]),
        "p1_prediction": float(quantiles[2]),
        "p5_prediction": float(quantiles[3]),
        "median_prediction": float(quantiles[4]),
        "p95_prediction": float(quantiles[5]),
        "p99_prediction": float(quantiles[6]),
        "p99_9_prediction": float(quantiles[7]),
        "max_prediction": float(quantiles[8]),
        "count_below_-0.005": int((values < -0.005).sum()),
        "count_below_-0.01": int((values < -0.01).sum()),
        "count_above_0.01": int((values > 0.01).sum()),
    }


def _fit_one(X_train: np.ndarray, y_train: np.ndarray, X_holdout: np.ndarray, loss_name: str):
    np.random.seed(baseline.SEED)
    torch.manual_seed(baseline.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(baseline.SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    loader_generator = torch.Generator().manual_seed(baseline.SEED)
    loader = DataLoader(dataset, batch_size=baseline.BATCH_SIZE, shuffle=True, generator=loader_generator)
    model = baseline.TimeSeriesTransformer(baseline.D_MODEL, baseline.D_FF, baseline.N_BLOCKS).to(device)
    loss_fn = baseline.make_loss(loss_name)
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
            print(f"{loss_name} epoch {epoch + 1:3d}/{baseline.EPOCHS}: loss={history[-1]:.8f}")
    model.eval()
    with torch.inference_mode():
        prediction = model(torch.from_numpy(X_holdout).to(device)).cpu().numpy().reshape(-1)
    return prediction, history, str(device), sum(parameter.numel() for parameter in model.parameters())


def _state_table(metadata: pd.DataFrame, actual: np.ndarray, predictions: dict[str, np.ndarray], ar1: np.ndarray, rw: np.ndarray) -> pd.DataFrame:
    rows = []
    for state_name, group in metadata.groupby("state_name", sort=True):
        indices = group.index.to_numpy()
        actual_state = actual[indices]
        row = {"state_name": state_name}
        for name, values in predictions.items():
            row[f"{name}_rmse"] = _metrics(actual_state, values[indices])["rmse"]
            row[f"{name}_mae"] = _metrics(actual_state, values[indices])["mae"]
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

    coefficients = fit_ar1_excluding(data)
    ar1_growth = baseline.forecast_ar1(data, coefficients, baseline.HOLDOUT_START)
    actual_growth = metadata["actual_log_diff"].to_numpy()
    previous_level = metadata["previous_level"].to_numpy()
    actual_level = metadata["actual_level"].to_numpy()
    rw_growth = np.zeros(len(metadata), dtype=float)
    ar1_level = baseline.invert_log_difference(ar1_growth, previous_level)
    rw_level = previous_level.copy()

    predictions: dict[str, np.ndarray] = {}
    histories: dict[str, list[float]] = {}
    device = ""
    parameter_count = 0
    for loss_name in LOSSES:
        prediction_scaled, history, device, parameter_count = _fit_one(X_train, y_train, X_holdout, loss_name)
        row_scaling = metadata[["state_fips"]].merge(scaling, on="state_fips", how="left", sort=False, validate="many_to_one")
        predictions[loss_name] = prediction_scaled * row_scaling["std"].to_numpy() + row_scaling["mean"].to_numpy()
        histories[loss_name] = history

    levels = {
        name: baseline.invert_log_difference(values, previous_level) for name, values in predictions.items()
    }
    state_table = _state_table(metadata, actual_level, levels, ar1_level, rw_level)
    state_table.to_csv(STATE_RESULTS_PATH, index=False)

    rows = []
    for loss_name in LOSSES:
        growth_metrics = _metrics(actual_growth, predictions[loss_name])
        level_metrics = _metrics(actual_level, levels[loss_name])
        level_sse = (levels[loss_name] - actual_level) ** 2
        order = np.argsort(level_sse)[::-1]
        rows.append(
            {
                "loss": loss_name,
                "window": WINDOW_LENGTH,
                "training_windows": len(X_train),
                "growth_rmse": growth_metrics["rmse"],
                "growth_mae": growth_metrics["mae"],
                "level_rmse": level_metrics["rmse"],
                "level_mae": level_metrics["mae"],
                "median_state_rmse": state_table[f"{loss_name}_rmse"].median(),
                "states_beating_ar1": int((state_table[f"{loss_name}_rmse"] < state_table.ar1_rmse).sum()),
                "states_beating_rw": int((state_table[f"{loss_name}_rmse"] < state_table.rw_rmse).sum()),
                **_tail(predictions[loss_name]),
                "top1_sse_share": level_sse[order[:1]].sum() / level_sse.sum(),
                "top5_sse_share": level_sse[order[:5]].sum() / level_sse.sum(),
                "top10_sse_share": level_sse[order[:10]].sum() / level_sse.sum(),
                "top20_sse_share": level_sse[order[:20]].sum() / level_sse.sum(),
                "max_abs_level_error": np.abs(levels[loss_name] - actual_level).max(),
                "max_error_state": metadata.iloc[np.argmax(np.abs(levels[loss_name] - actual_level))]["state_name"],
                "max_error_date": metadata.iloc[np.argmax(np.abs(levels[loss_name] - actual_level))]["forecast_date"].strftime("%Y-%m"),
                "initial_loss": histories[loss_name][0],
                "final_loss": histories[loss_name][-1],
                "minimum_loss": min(histories[loss_name]),
            }
        )
    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_PATH, index=False)

    yearly = []
    for year in [2023, 2024, 2025, 2026]:
        mask = metadata["forecast_date"].dt.year.to_numpy() == year
        row = {"year": year}
        for loss_name in LOSSES:
            row[f"{loss_name}_rmse"] = _metrics(actual_level[mask], levels[loss_name][mask])["rmse"]
        yearly.append(row)
    yearly_df = pd.DataFrame(yearly)
    baseline_diag = pd.read_csv(baseline.DIAGNOSTICS_PATH, parse_dates=["forecast_date"])
    benchmark_growth = {name: _metrics(baseline_diag.actual_log_diff, baseline_diag[f"{name}_log_diff"]) for name in ["ar1", "rw"]}
    benchmark_level = {name: _metrics(baseline_diag.actual_level, baseline_diag[f"{name}_level"]) for name in ["ar1", "rw"]}

    summary = [
        "CES State total-nonfarm MSE versus Huber loss comparison",
        f"window_length: {WINDOW_LENGTH}",
        f"covid_exclusion: {COVID_EXCLUDE_START} through {COVID_EXCLUDE_END}",
        f"train_end: {baseline.TRAIN_END}",
        f"holdout: {baseline.HOLDOUT_START} through {holdout_rows.date.max().strftime('%Y-%m')}",
        f"constants: d_model={baseline.D_MODEL}, d_ff={baseline.D_FF}, n_blocks={baseline.N_BLOCKS}, batch_size={baseline.BATCH_SIZE}, learning_rate={baseline.LEARNING_RATE}, epochs={baseline.EPOCHS}, seed={baseline.SEED}",
        f"device: {device}",
        f"transformer_parameters: {parameter_count}",
        f"training_windows: {len(X_train)}",
        "benchmark_growth_metrics:",
        *(f"  {name}: {value}" for name, value in benchmark_growth.items()),
        "benchmark_level_metrics:",
        *(f"  {name}: {value}" for name, value in benchmark_level.items()),
        "loss_results:",
        *[f"  {row.to_dict()}" for _, row in results.iterrows()],
        "yearly_level_rmse:",
        *[f"  {row.to_dict()}" for _, row in yearly_df.iterrows()],
        f"huber_beats_mse_states: {state_table.loc[state_table.huber_rmse < state_table.mse_rmse, 'state_name'].tolist()}",
        f"mse_beats_huber_states: {state_table.loc[state_table.mse_rmse < state_table.huber_rmse, 'state_name'].tolist()}",
        f"ties: {state_table.loc[np.isclose(state_table.mse_rmse, state_table.huber_rmse), 'state_name'].tolist()}",
        "largest_absolute_growth_predictions:",
    ]
    for loss_name in LOSSES:
        summary.append(f"  {loss_name}:")
        diagnostic = metadata[["state_name", "forecast_date"]].copy()
        diagnostic["prediction"] = predictions[loss_name]
        diagnostic["abs_prediction"] = np.abs(predictions[loss_name])
        for _, row in diagnostic.nlargest(10, "abs_prediction").iterrows():
            summary.append(f"    {row.to_dict()}")
    summary.append("sample_forecasts:")
    for state, date_text in [("Texas", "2024-08"), ("Florida", "2024-11"), ("Washington", "2024-11")]:
        row_index = metadata.index[(metadata.state_name == state) & (metadata.forecast_date == pd.Timestamp(f"{date_text}-01"))][0]
        summary.append(str({
            "state": state,
            "date": date_text,
            "actual_growth": actual_growth[row_index],
            "mse_growth": predictions["mse"][row_index],
            "huber_growth": predictions["huber"][row_index],
            "ar1_growth": ar1_growth[row_index],
            "actual_level": actual_level[row_index],
            "mse_level_error": levels["mse"][row_index] - actual_level[row_index],
            "huber_level_error": levels["huber"][row_index] - actual_level[row_index],
        }))
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
