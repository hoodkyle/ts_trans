"""Run a chronological CES state holdout experiment.

This is one deliberately explicit experiment rather than a forecasting
framework.  A shared tiny transformer is compared with per-state AR(1) and
last-value forecasts on monthly log differences.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from ts_trans.model.forecast import TimeSeriesTransformer
from ts_trans.training import make_loss


WINDOW_LENGTH = 12
D_MODEL = 8
D_FF = 16
N_BLOCKS = 2
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
EPOCHS = 150
SEED = 20260824
LOSS = "mse"
TRAIN_END = "2022-12"
HOLDOUT_START = "2023-01"


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "real" / "ces_state_total_nonfarm.csv"
SUMMARY_PATH = ROOT / "output" / "train_ces_state_holdout_summary.txt"
STATE_METRICS_PATH = ROOT / "output" / "train_ces_state_holdout_by_state.csv"
DIAGNOSTICS_PATH = ROOT / "output" / "train_ces_state_holdout_diagnostics.csv"


def _metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = np.asarray(prediction, dtype=float) - np.asarray(actual, dtype=float)
    mse = float(np.mean(error**2))
    return {"mse": mse, "rmse": float(np.sqrt(mse)), "mae": float(np.mean(np.abs(error)))}


def make_log_difference(data: pd.DataFrame) -> pd.DataFrame:
    """Add level, log-level, and within-state monthly log-difference columns."""
    required = {"state_fips", "state_name", "series_id", "date", "value"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"CES data is missing required columns: {missing}")
    result = data.copy(deep=True)
    result["date"] = pd.to_datetime(result["date"], format="%Y-%m", errors="coerce")
    if result["date"].isna().any():
        raise ValueError("CES date values must be valid YYYY-MM months")
    result["level"] = pd.to_numeric(result["value"], errors="coerce")
    if result["level"].isna().any() or not np.isfinite(result["level"].to_numpy()).all():
        raise ValueError("CES employment levels must be finite numeric values")
    if (result["level"] <= 0).any():
        raise ValueError("CES employment levels must be strictly positive before taking logs")
    result = result.sort_values(["state_fips", "date"], kind="mergesort").reset_index(drop=True)
    result["log_level"] = np.log(result["level"])
    result["log_difference"] = result.groupby("state_fips", sort=False)["log_level"].diff()
    return result


def load_and_validate_ces(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load local CES data and verify 51 continuous monthly positive series."""
    data = pd.read_csv(path, dtype={"state_fips": "string"})
    transformed = make_log_difference(data)
    if transformed["series_id"].nunique() != 51 or transformed["state_fips"].nunique() != 51:
        raise ValueError("expected exactly 51 state/DC CES series")
    if transformed.duplicated(["state_fips", "date"]).any():
        raise ValueError("CES data contains duplicate state/month observations")
    for state_fips, group in transformed.groupby("state_fips", sort=False):
        dates = group["date"].sort_values()
        expected = pd.date_range(dates.iloc[0], dates.iloc[-1], freq="MS")
        if len(dates) != len(expected) or not dates.reset_index(drop=True).equals(pd.Series(expected)):
            raise ValueError(f"CES series {state_fips} is not continuous monthly data")
    return transformed


def chronological_split(data: pd.DataFrame, train_end: str = TRAIN_END, holdout_start: str = HOLDOUT_START):
    """Return train and holdout rows using date comparisons only."""
    train_end_date = pd.Period(train_end, freq="M").to_timestamp()
    holdout_start_date = pd.Period(holdout_start, freq="M").to_timestamp()
    if train_end_date >= holdout_start_date:
        raise ValueError("train_end must precede holdout_start")
    train = data[data["date"] <= train_end_date].copy()
    holdout = data[data["date"] >= holdout_start_date].copy()
    if train.empty or holdout.empty:
        raise ValueError("chronological split produced an empty partition")
    return train, holdout


def fit_state_scaling(train_data: pd.DataFrame) -> pd.DataFrame:
    """Fit per-state population mean/std using training log differences only."""
    rows = []
    for state_fips, group in train_data.groupby("state_fips", sort=True):
        values = group["log_difference"].dropna().to_numpy(dtype=float)
        if len(values) == 0:
            raise ValueError(f"state {state_fips} has no training log differences")
        mean = float(values.mean())
        std = float(values.std(ddof=0))
        if not np.isfinite(std) or std <= 1e-12:
            raise ValueError(f"state {state_fips} has degenerate training log differences")
        rows.append({"state_fips": state_fips, "mean": mean, "std": std})
    return pd.DataFrame(rows).sort_values("state_fips", kind="mergesort").reset_index(drop=True)


def apply_state_scaling(data: pd.DataFrame, scaling: pd.DataFrame) -> pd.DataFrame:
    """Apply already-fitted state scaling without refitting on the supplied data."""
    required = {"state_fips", "log_difference"}
    if not required.issubset(data.columns):
        raise ValueError("data must contain state_fips and log_difference")
    if scaling.duplicated("state_fips").any():
        raise ValueError("scaling must contain one row per state")
    result = data.copy(deep=True)
    matched = result[["state_fips"]].merge(scaling, on="state_fips", how="left", sort=False, validate="many_to_one")
    if matched[["mean", "std"]].isna().any().any():
        raise ValueError("scaling is missing a requested state")
    result["scaled_log_difference"] = (result["log_difference"] - matched["mean"]) / matched["std"]
    return result


def build_windows(data: pd.DataFrame, target_start: str, target_end: str | None = None):
    """Build one-step windows whose target dates lie in the requested range."""
    start = pd.Period(target_start, freq="M").to_timestamp()
    end = pd.Period(target_end, freq="M").to_timestamp() if target_end else None
    inputs: list[np.ndarray] = []
    targets: list[float] = []
    rows: list[dict[str, object]] = []
    for _, group in data.groupby("state_fips", sort=True):
        group = group.sort_values("date", kind="mergesort").reset_index(drop=True)
        values = group["scaled_log_difference"].to_numpy(dtype=float)
        for index in range(WINDOW_LENGTH, len(group)):
            target_date = group.loc[index, "date"]
            if target_date < start or (end is not None and target_date > end):
                continue
            window = values[index - WINDOW_LENGTH : index]
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
    metadata = pd.DataFrame(rows)
    return np.asarray(inputs, dtype=np.float32), np.asarray(targets, dtype=np.float32)[:, None], metadata


def fit_ar1(train_data: pd.DataFrame) -> pd.DataFrame:
    """Fit per-state OLS g_t = alpha + rho*g_(t-1) on training rows only."""
    rows = []
    for state_fips, group in train_data.groupby("state_fips", sort=True):
        group = group.sort_values("date", kind="mergesort")
        values = group["log_difference"].to_numpy(dtype=float)
        valid = np.isfinite(values[1:]) & np.isfinite(values[:-1])
        y = values[1:][valid]
        lag = values[:-1][valid]
        if len(y) < 2:
            raise ValueError(f"state {state_fips} has too few observations for AR(1)")
        alpha, rho = np.linalg.lstsq(np.column_stack([np.ones(len(lag)), lag]), y, rcond=None)[0]
        rows.append({"state_fips": state_fips, "alpha": float(alpha), "rho": float(rho)})
    return pd.DataFrame(rows).sort_values("state_fips", kind="mergesort").reset_index(drop=True)


def forecast_ar1(
    data: pd.DataFrame, coefficients: pd.DataFrame, target_start: str = HOLDOUT_START
) -> np.ndarray:
    """Forecast target dates using actual prior-month growth from full history."""
    ordered = data.sort_values(["state_fips", "date"], kind="mergesort").copy()
    ordered["previous_growth"] = ordered.groupby("state_fips", sort=False)["log_difference"].shift(1)
    ordered = ordered[ordered["date"] >= pd.Period(target_start, freq="M").to_timestamp()]
    matched = ordered[["state_fips"]].merge(coefficients, on="state_fips", how="left", sort=False, validate="many_to_one")
    if matched[["alpha", "rho"]].isna().any().any():
        raise ValueError("AR(1) coefficients are missing a holdout state")
    if ordered["previous_growth"].isna().any():
        raise ValueError("a holdout forecast is missing its known prior-month growth")
    result = matched["alpha"].to_numpy() + matched["rho"].to_numpy() * ordered["previous_growth"].to_numpy()
    return result


def invert_log_difference(growth: np.ndarray, previous_level: np.ndarray) -> np.ndarray:
    """Convert forecast log differences back to levels using known prior levels."""
    growth = np.asarray(growth, dtype=float)
    previous_level = np.asarray(previous_level, dtype=float)
    if growth.shape != previous_level.shape:
        raise ValueError("growth and previous_level must have the same shape")
    if not np.isfinite(growth).all() or not np.isfinite(previous_level).all() or (previous_level <= 0).any():
        raise ValueError("growth and previous levels must be finite, with positive previous levels")
    return previous_level * np.exp(growth)


def _state_level_metrics(metadata: pd.DataFrame, predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for state_name, group in metadata.groupby("state_name", sort=True):
        indices = group.index.to_numpy()
        actual = group["actual_level"].to_numpy()
        row: dict[str, object] = {"state_name": state_name, "n_holdout": len(group)}
        for name, values in predictions.items():
            row[f"{name}_rmse"] = _metrics(actual, values[indices])["rmse"]
            row[f"{name}_mae"] = _metrics(actual, values[indices])["mae"]
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    data = load_and_validate_ces()
    train_rows, holdout_rows = chronological_split(data)
    scaling = fit_state_scaling(train_rows)
    scaled = apply_state_scaling(data, scaling)
    X_train, y_train, _ = build_windows(scaled, "2005-01", TRAIN_END)
    X_holdout, y_holdout, metadata = build_windows(scaled, HOLDOUT_START)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    loader_generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, generator=loader_generator)
    model = TimeSeriesTransformer(D_MODEL, D_FF, N_BLOCKS).to(device)
    loss_fn = make_loss(LOSS)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    epoch_history: list[float] = []

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        total_items = 0
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            prediction = model(batch_X)
            loss = loss_fn(prediction, batch_y)
            loss.backward()
            optimizer.step()
            count = len(batch_X)
            total_loss += loss.item() * count
            total_items += count
        epoch_loss = total_loss / total_items
        epoch_history.append(epoch_loss)
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1:
            print(f"epoch {epoch + 1:3d}/{EPOCHS}: loss={epoch_loss:.8f}")

    model.eval()
    with torch.inference_mode():
        transformer_scaled = model(torch.from_numpy(X_holdout).to(device)).cpu().numpy().reshape(-1)
    scaling_by_row = metadata[["state_fips"]].merge(scaling, on="state_fips", how="left", sort=False, validate="many_to_one")
    transformer_growth = transformer_scaled * scaling_by_row["std"].to_numpy() + scaling_by_row["mean"].to_numpy()
    coefficients = fit_ar1(train_rows)
    ar1_order = metadata.sort_values(["state_fips", "forecast_date"], kind="mergesort")
    ar1_growth_ordered = forecast_ar1(data, coefficients, HOLDOUT_START)
    ar1_growth = pd.Series(ar1_growth_ordered, index=ar1_order.index).reindex(metadata.index).to_numpy()
    rw_growth = np.zeros(len(metadata), dtype=float)
    previous_level = metadata["previous_level"].to_numpy()
    actual_level = metadata["actual_level"].to_numpy()
    actual_growth = metadata["actual_log_diff"].to_numpy()
    level_predictions = {
        "transformer": invert_log_difference(transformer_growth, previous_level),
        "ar1": invert_log_difference(ar1_growth, previous_level),
        "rw": invert_log_difference(rw_growth, previous_level),
    }
    growth_predictions = {"transformer": transformer_growth, "ar1": ar1_growth, "rw": rw_growth}
    growth_metrics = {name: _metrics(actual_growth, values) for name, values in growth_predictions.items()}
    level_metrics = {name: _metrics(actual_level, values) for name, values in level_predictions.items()}
    state_metrics = _state_level_metrics(metadata, level_predictions)
    state_metrics.to_csv(STATE_METRICS_PATH, index=False)
    year_rows = []
    years = metadata["forecast_date"].dt.year
    for year, indices in metadata.groupby(years).groups.items():
        actual = actual_level[indices]
        year_rows.append({"year": int(year), **{f"{name}_rmse": _metrics(actual, values[indices])["rmse"] for name, values in level_predictions.items()}})
    year_metrics = pd.DataFrame(year_rows)
    alpha_summary = coefficients["alpha"].describe()
    rho_summary = coefficients["rho"].describe()
    summary_lines = [
        "CES state total nonfarm chronological holdout experiment",
        f"train_period: 2005-01 through {TRAIN_END}",
        f"holdout_period: {HOLDOUT_START} through {metadata['forecast_date'].max().strftime('%Y-%m')}",
        "holdout_design: rolling one-step forecasts using actual prior-month observations",
        "covid_note: training includes 2020-2022; no shock periods were removed",
        f"states: {data['state_fips'].nunique()}",
        f"training_windows: {len(X_train)}",
        f"holdout_forecasts: {len(X_holdout)}",
        f"device: {device}",
        f"window_length: {WINDOW_LENGTH}",
        f"d_model: {D_MODEL}",
        f"d_ff: {D_FF}",
        f"n_blocks: {N_BLOCKS}",
        f"batch_size: {BATCH_SIZE}",
        f"learning_rate: {LEARNING_RATE}",
        f"epochs: {EPOCHS}",
        f"seed: {SEED}",
        f"loss: {LOSS}",
        f"transformer_parameters: {parameter_count}",
        f"initial_training_loss: {epoch_history[0]:.10f}",
        f"final_training_loss: {epoch_history[-1]:.10f}",
        f"minimum_training_loss: {min(epoch_history):.10f}",
        "growth_metrics:",
        *(f"  {name}: {json.dumps(values)}" for name, values in growth_metrics.items()),
        "level_metrics:",
        *(f"  {name}: {json.dumps(values)}" for name, values in level_metrics.items()),
        f"states_transformer_beats_ar1_rmse: {int((state_metrics['transformer_rmse'] < state_metrics['ar1_rmse']).sum())}",
        f"states_transformer_beats_rw_rmse: {int((state_metrics['transformer_rmse'] < state_metrics['rw_rmse']).sum())}",
        "median_state_rmse:",
        *(f"  {name}: {state_metrics[f'{name}_rmse'].median():.10f}" for name in level_predictions),
        "ar1_alpha_summary:",
        *(f"  {key}: {value:.10f}" for key, value in alpha_summary.items()),
        "ar1_rho_summary:",
        *(f"  {key}: {value:.10f}" for key, value in rho_summary.items()),
        "year_level_rmse:",
        *[f"  {row.to_dict()}" for _, row in year_metrics.iterrows()],
        "epoch_loss_history:",
        *[f"  epoch_{index + 1:03d}: {loss:.10f}" for index, loss in enumerate(epoch_history)],
        "sample_forecasts:",
    ]
    sample = metadata.copy()
    sample["transformer_log_diff"] = transformer_growth
    sample["ar1_log_diff"] = ar1_growth
    sample["transformer_level"] = level_predictions["transformer"]
    sample["ar1_level"] = level_predictions["ar1"]
    sample["rw_level"] = level_predictions["rw"]
    sample["transformer_level_error"] = sample["transformer_level"] - sample["actual_level"]
    sample["ar1_level_error"] = sample["ar1_level"] - sample["actual_level"]
    sample["rw_level_error"] = sample["rw_level"] - sample["actual_level"]
    sample = sample.sort_values(["state_fips", "forecast_date"], kind="mergesort").head(5)
    diagnostics = metadata.copy()
    diagnostics["rw_log_diff"] = rw_growth
    diagnostics["transformer_log_diff"] = transformer_growth
    diagnostics["ar1_log_diff"] = ar1_growth
    diagnostics["rw_level"] = level_predictions["rw"]
    diagnostics["transformer_level"] = level_predictions["transformer"]
    diagnostics["ar1_level"] = level_predictions["ar1"]
    diagnostics["transformer_growth_error"] = transformer_growth - actual_growth
    diagnostics["ar1_growth_error"] = ar1_growth - actual_growth
    diagnostics["rw_growth_error"] = rw_growth - actual_growth
    diagnostics["transformer_level_error"] = level_predictions["transformer"] - actual_level
    diagnostics["ar1_level_error"] = level_predictions["ar1"] - actual_level
    diagnostics["rw_level_error"] = level_predictions["rw"] - actual_level
    diagnostics["train_mean_log_diff"] = scaling_by_row["mean"].to_numpy()
    diagnostics["train_std_log_diff"] = scaling_by_row["std"].to_numpy()
    diagnostics["transformer_scaled_log_diff"] = transformer_scaled
    diagnostics = diagnostics.sort_values(["state_fips", "forecast_date"], kind="mergesort")
    diagnostics.to_csv(DIAGNOSTICS_PATH, index=False)
    sample_columns = [
        "state_name", "forecast_date", "previous_level", "actual_level", "actual_log_diff",
        "transformer_log_diff", "ar1_log_diff", "transformer_level", "ar1_level", "rw_level",
        "transformer_level_error", "ar1_level_error", "rw_level_error",
    ]
    for _, row in sample[sample_columns].iterrows():
        summary_lines.append(str(row.to_dict()))
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"device: {device}")
    print(f"training windows: {len(X_train)}; holdout forecasts: {len(X_holdout)}")
    print(f"transformer parameters: {parameter_count}")
    print(f"initial/final/minimum loss: {epoch_history[0]:.8f}/{epoch_history[-1]:.8f}/{min(epoch_history):.8f}")
    print(f"growth metrics: {growth_metrics}")
    print(f"level metrics: {level_metrics}")
    print(f"states won vs AR(1)/RW: {int((state_metrics['transformer_rmse'] < state_metrics['ar1_rmse']).sum())}/{int((state_metrics['transformer_rmse'] < state_metrics['rw_rmse']).sum())}")
    print(f"median state RMSE: {state_metrics[['transformer_rmse', 'ar1_rmse', 'rw_rmse']].median().to_dict()}")
    print(f"AR(1) alpha mean/range: {alpha_summary['mean']:.8f}/{alpha_summary['min']:.8f}..{alpha_summary['max']:.8f}")
    print(f"AR(1) rho mean/range: {rho_summary['mean']:.8f}/{rho_summary['min']:.8f}..{rho_summary['max']:.8f}")
    print("year diagnostics:")
    print(year_metrics.to_string(index=False))
    print("sample forecasts:")
    print(sample[sample_columns].to_string(index=False))
    print(f"summary: {SUMMARY_PATH}")
    print(f"state metrics: {STATE_METRICS_PATH}")
    print(f"diagnostics: {DIAGNOSTICS_PATH}")


if __name__ == "__main__":
    main()
