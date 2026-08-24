"""Run the fixed CES holdout experiment with COVID targets excluded."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

# Reuse the baseline's model, data checks, windowing, metrics, and inversion.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_ces_state_holdout as baseline


COVID_EXCLUDE_START = "2020-03"
COVID_EXCLUDE_END = "2021-12"

SUMMARY_PATH = baseline.ROOT / "output" / "train_ces_state_holdout_no_covid_summary.txt"
STATE_METRICS_PATH = baseline.ROOT / "output" / "train_ces_state_holdout_no_covid_by_state.csv"
DIAGNOSTICS_PATH = baseline.ROOT / "output" / "train_ces_state_holdout_no_covid_diagnostics.csv"


def excluded_target_mask(dates: pd.Series | pd.DatetimeIndex) -> pd.Series:
    """Return the explicit COVID target-date exclusion mask."""
    dates = pd.Series(pd.to_datetime(dates))
    start = pd.Period(COVID_EXCLUDE_START, freq="M").to_timestamp()
    end = pd.Period(COVID_EXCLUDE_END, freq="M").to_timestamp()
    return dates.between(start, end)


def eligible_training_rows(data: pd.DataFrame) -> pd.DataFrame:
    """Keep estimation rows through TRAIN_END except excluded target months."""
    train_end = pd.Period(baseline.TRAIN_END, freq="M").to_timestamp()
    mask = (data["date"] <= train_end) & ~excluded_target_mask(data["date"])
    return data.loc[mask].copy()


def filter_training_windows(X: np.ndarray, y: np.ndarray, metadata: pd.DataFrame):
    """Apply target-date exclusion to already-constructed training windows."""
    keep = ~excluded_target_mask(metadata["forecast_date"])
    return X[keep.to_numpy()], y[keep.to_numpy()], metadata.loc[keep].reset_index(drop=True)


def count_retained_windows_overlapping_exclusion(data: pd.DataFrame) -> int:
    """Count eligible target windows whose 12-month input contains COVID dates."""
    count = 0
    train_end = pd.Period(baseline.TRAIN_END, freq="M").to_timestamp()
    for _, group in data.groupby("state_fips", sort=True):
        group = group.sort_values("date", kind="mergesort").reset_index(drop=True)
        for index in range(baseline.WINDOW_LENGTH, len(group)):
            target_date = group.loc[index, "date"]
            if target_date > train_end or excluded_target_mask(pd.Series([target_date])).iloc[0]:
                continue
            input_dates = group.loc[index - baseline.WINDOW_LENGTH : index - 1, "date"]
            if excluded_target_mask(input_dates).any():
                count += 1
    return count


def fit_ar1_excluding(data: pd.DataFrame) -> pd.DataFrame:
    """Fit AR(1) only for eligible target rows, using actual prior observations."""
    rows = []
    train_end = pd.Period(baseline.TRAIN_END, freq="M").to_timestamp()
    for state_fips, group in data.groupby("state_fips", sort=True):
        group = group.sort_values("date", kind="mergesort").reset_index(drop=True)
        values = group["log_difference"].to_numpy(dtype=float)
        dates = group["date"]
        current_is_eligible = (dates <= train_end).to_numpy() & ~excluded_target_mask(dates).to_numpy()
        valid = current_is_eligible[1:] & np.isfinite(values[1:]) & np.isfinite(values[:-1])
        y = values[1:][valid]
        lag = values[:-1][valid]
        if len(y) < 2:
            raise ValueError(f"state {state_fips} has too few eligible observations for AR(1)")
        alpha, rho = np.linalg.lstsq(np.column_stack([np.ones(len(lag)), lag]), y, rcond=None)[0]
        rows.append({"state_fips": state_fips, "alpha": float(alpha), "rho": float(rho)})
    return pd.DataFrame(rows).sort_values("state_fips", kind="mergesort").reset_index(drop=True)


def _quantiles(values: pd.Series | np.ndarray) -> dict[str, float]:
    probabilities = [0, 0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999, 1]
    array = np.asarray(values, dtype=float)
    return {str(probability): float(value) for probability, value in zip(probabilities, np.quantile(array, probabilities))}


def _row_keys(data: pd.DataFrame) -> list[tuple[str, str]]:
    """Compare state/month keys without depending on CSV dtype inference."""
    dates = pd.to_datetime(data["forecast_date"]).dt.strftime("%Y-%m")
    state_fips = data["state_fips"].astype(str).str.zfill(2)
    return list(zip(state_fips, dates))


def _state_metrics(metadata: pd.DataFrame, levels: dict[str, np.ndarray], growth_errors: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for state_name, group in metadata.groupby("state_name", sort=True):
        indices = group.index.to_numpy()
        actual = group["actual_level"].to_numpy()
        row = {"state_name": state_name, "n_holdout": len(group)}
        for name, values in levels.items():
            row[f"{name}_rmse"] = baseline._metrics(actual, values[indices])["rmse"]
            row[f"{name}_mae"] = baseline._metrics(actual, values[indices])["mae"]
        row["transformer_growth_rmse"] = baseline._metrics(
            np.zeros(len(indices)), growth_errors["transformer"][indices]
        )["rmse"]
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    np.random.seed(baseline.SEED)
    torch.manual_seed(baseline.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(baseline.SEED)

    data = baseline.load_and_validate_ces()
    train_rows, holdout_rows = baseline.chronological_split(data)
    eligible_rows = eligible_training_rows(data)
    baseline_scaling = baseline.fit_state_scaling(train_rows)
    scaling = baseline.fit_state_scaling(eligible_rows)
    scaled = baseline.apply_state_scaling(data, scaling)
    X_all, y_all, all_metadata = baseline.build_windows(scaled, "2005-01", baseline.TRAIN_END)
    X_train, y_train, train_metadata = filter_training_windows(X_all, y_all, all_metadata)
    X_holdout, y_holdout, metadata = baseline.build_windows(scaled, baseline.HOLDOUT_START)

    baseline_scaled = baseline.apply_state_scaling(data, baseline_scaling)
    X_baseline, _, baseline_holdout_metadata = baseline.build_windows(
        baseline_scaled, baseline.HOLDOUT_START
    )
    if not metadata[["state_fips", "forecast_date"]].reset_index(drop=True).equals(
        baseline_holdout_metadata[["state_fips", "forecast_date"]].reset_index(drop=True)
    ):
        raise RuntimeError("no-COVID and baseline holdout rows are not identical")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    loader_generator = torch.Generator().manual_seed(baseline.SEED)
    loader = DataLoader(dataset, batch_size=baseline.BATCH_SIZE, shuffle=True, generator=loader_generator)
    model = baseline.TimeSeriesTransformer(baseline.D_MODEL, baseline.D_FF, baseline.N_BLOCKS).to(device)
    loss_fn = baseline.make_loss(baseline.LOSS)
    optimizer = torch.optim.AdamW(model.parameters(), lr=baseline.LEARNING_RATE)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
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
        transformer_scaled = model(torch.from_numpy(X_holdout).to(device)).cpu().numpy().reshape(-1)
    scaling_by_row = metadata[["state_fips"]].merge(scaling, on="state_fips", how="left", sort=False, validate="many_to_one")
    transformer_growth = transformer_scaled * scaling_by_row["std"].to_numpy() + scaling_by_row["mean"].to_numpy()
    coefficients = fit_ar1_excluding(data)
    ar1_growth = baseline.forecast_ar1(data, coefficients, baseline.HOLDOUT_START)
    actual_growth = metadata["actual_log_diff"].to_numpy()
    previous_level = metadata["previous_level"].to_numpy()
    actual_level = metadata["actual_level"].to_numpy()
    rw_growth = np.zeros(len(metadata), dtype=float)
    growth = {"transformer": transformer_growth, "ar1": ar1_growth, "rw": rw_growth}
    levels = {name: baseline.invert_log_difference(values, previous_level) for name, values in growth.items()}
    growth_errors = {name: values - actual_growth for name, values in growth.items()}
    level_errors = {name: values - actual_level for name, values in levels.items()}
    growth_metrics = {name: baseline._metrics(actual_growth, values) for name, values in growth.items()}
    level_metrics = {name: baseline._metrics(actual_level, values) for name, values in levels.items()}

    state_metrics = _state_metrics(metadata, levels, growth_errors)
    STATE_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    state_metrics.to_csv(STATE_METRICS_PATH, index=False)
    diagnostics = metadata.copy()
    diagnostics["rw_log_diff"] = rw_growth
    diagnostics["transformer_log_diff"] = transformer_growth
    diagnostics["ar1_log_diff"] = ar1_growth
    for name in levels:
        diagnostics[f"{name}_level"] = levels[name]
        diagnostics[f"{name}_growth_error"] = growth_errors[name]
        diagnostics[f"{name}_level_error"] = level_errors[name]
    diagnostics["train_mean_log_diff"] = scaling_by_row["mean"].to_numpy()
    diagnostics["train_std_log_diff"] = scaling_by_row["std"].to_numpy()
    diagnostics["transformer_scaled_log_diff"] = transformer_scaled
    diagnostics = diagnostics.sort_values(["state_fips", "forecast_date"], kind="mergesort").reset_index(drop=True)
    diagnostics.to_csv(DIAGNOSTICS_PATH, index=False)

    baseline_diagnostics = pd.read_csv(baseline.DIAGNOSTICS_PATH, parse_dates=["forecast_date"])
    if _row_keys(baseline_diagnostics) != _row_keys(diagnostics):
        raise RuntimeError("baseline and no-COVID diagnostics do not use identical rows")
    baseline_metrics = {
        "growth": {name: baseline._metrics(baseline_diagnostics.actual_log_diff, baseline_diagnostics[f"{name}_log_diff"]) for name in ["transformer", "ar1", "rw"]},
        "level": {name: baseline._metrics(baseline_diagnostics.actual_level, baseline_diagnostics[f"{name}_level"]) for name in ["transformer", "ar1", "rw"]},
    }

    train_values = eligible_rows["log_difference"].dropna()
    baseline_train_values = train_rows["log_difference"].dropna()
    holdout_values = holdout_rows["log_difference"].dropna()
    year = metadata["forecast_date"].dt.year
    year_bias = pd.DataFrame({f"{name}_{kind}_error": values for name in growth for kind, values in [("growth", growth_errors[name]), ("level", level_errors[name])]})
    year_bias["year"] = year.to_numpy()
    year_bias = year_bias.groupby("year").mean()
    covid_overlap = count_retained_windows_overlapping_exclusion(data)
    sse = level_errors["transformer"] ** 2
    twenty = np.argsort(sse)[::-1]
    year_2024 = metadata["forecast_date"].dt.year == 2024
    sse_2024 = sse[year_2024.to_numpy()]
    state_2024 = pd.DataFrame({"state_name": metadata.loc[year_2024, "state_name"].to_numpy(), "sse": sse_2024}).groupby("state_name").sse.sum().sort_values(ascending=False)

    summary = [
        "CES state holdout experiment excluding COVID target months",
        f"covid_exclude: {COVID_EXCLUDE_START} through {COVID_EXCLUDE_END}",
        f"train_period: 2005-01 through {baseline.TRAIN_END}",
        f"holdout_period: {baseline.HOLDOUT_START} through {metadata.forecast_date.max().strftime('%Y-%m')}",
        f"states: {data.state_fips.nunique()}",
        f"baseline_training_windows: {len(X_all)}",
        f"no_covid_training_windows: {len(X_train)}",
        f"retained_windows_with_covid_input_overlap: {covid_overlap}",
        f"holdout_forecasts: {len(X_holdout)}",
        f"device: {device}",
        f"transformer_parameters: {parameter_count}",
        f"window_length: {baseline.WINDOW_LENGTH}",
        f"d_model: {baseline.D_MODEL}",
        f"d_ff: {baseline.D_FF}",
        f"n_blocks: {baseline.N_BLOCKS}",
        f"batch_size: {baseline.BATCH_SIZE}",
        f"learning_rate: {baseline.LEARNING_RATE}",
        f"epochs: {baseline.EPOCHS}",
        f"seed: {baseline.SEED}",
        f"initial_training_loss: {history[0]:.10f}",
        f"final_training_loss: {history[-1]:.10f}",
        f"minimum_training_loss: {min(history):.10f}",
        "no_covid_growth_metrics:",
        *(f"  {name}: {values}" for name, values in growth_metrics.items()),
        "no_covid_level_metrics:",
        *(f"  {name}: {values}" for name, values in level_metrics.items()),
        "baseline_growth_metrics:",
        *(f"  {name}: {values}" for name, values in baseline_metrics["growth"].items()),
        "baseline_level_metrics:",
        *(f"  {name}: {values}" for name, values in baseline_metrics["level"].items()),
        f"states_transformer_beats_ar1: {int((state_metrics.transformer_rmse < state_metrics.ar1_rmse).sum())}",
        f"states_transformer_beats_rw: {int((state_metrics.transformer_rmse < state_metrics.rw_rmse).sum())}",
        "median_state_rmse:",
        *(f"  {name}: {state_metrics[f'{name}_rmse'].median():.10f}" for name in levels),
        "no_covid_transformer_growth_quantiles:",
        str(_quantiles(transformer_growth)),
        f"transformer_predictions_below_-0.01: {int((transformer_growth < -0.01).sum())}",
        f"transformer_predictions_below_-0.02: {int((transformer_growth < -0.02).sum())}",
        f"transformer_predictions_below_-0.05: {int((transformer_growth < -0.05).sum())}",
        f"most_negative_transformer_prediction: {transformer_growth.min():.10f}",
        "training_distribution:",
        f"  baseline: mean={baseline_train_values.mean():.10f}, std={baseline_train_values.std(ddof=0):.10f}, min={baseline_train_values.min():.10f}, max={baseline_train_values.max():.10f}, q01={baseline_train_values.quantile(.01):.10f}, q99={baseline_train_values.quantile(.99):.10f}",
        f"  no_covid: mean={train_values.mean():.10f}, std={train_values.std(ddof=0):.10f}, min={train_values.min():.10f}, max={train_values.max():.10f}, q01={train_values.quantile(.01):.10f}, q99={train_values.quantile(.99):.10f}",
        f"  holdout: mean={holdout_values.mean():.10f}, std={holdout_values.std(ddof=0):.10f}, min={holdout_values.min():.10f}, max={holdout_values.max():.10f}, q01={holdout_values.quantile(.01):.10f}, q99={holdout_values.quantile(.99):.10f}",
        f"training_std_summary: min={scaling['std'].min():.10f}, median={scaling['std'].median():.10f}, max={scaling['std'].max():.10f}",
        f"worst_error_concentration: top1={sse[twenty[:1]].sum()/sse.sum():.6f}, top5={sse[twenty[:5]].sum()/sse.sum():.6f}, top10={sse[twenty[:10]].sum()/sse.sum():.6f}, top20={sse[twenty[:20]].sum()/sse.sum():.6f}",
        f"2024_transformer_rmse: {np.sqrt(sse_2024.mean()):.10f}",
        f"2024_state_sse_top10: {state_2024.head(10).to_dict()}",
        "year_bias:",
        str(year_bias),
        "sample_problem_rows:",
    ]
    for state, date_text in [("Texas", "2024-08"), ("Florida", "2024-11"), ("Washington", "2024-11")]:
        row = diagnostics[(diagnostics.state_name == state) & (diagnostics.forecast_date == pd.Timestamp(f"{date_text}-01"))].iloc[0]
        old = baseline_diagnostics[(baseline_diagnostics.state_name == state) & (baseline_diagnostics.forecast_date == pd.Timestamp(f"{date_text}-01"))].iloc[0]
        summary.append(str({"state": state, "date": date_text, "actual_growth": row.actual_log_diff, "baseline_transformer_growth": old.transformer_log_diff, "no_covid_transformer_growth": row.transformer_log_diff, "ar1_growth": row.ar1_log_diff, "actual_level": row.actual_level, "no_covid_transformer_level_error": row.transformer_level_error}))
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text("\n".join(summary) + "\n", encoding="utf-8")

    print(f"device: {device}; training windows baseline/no-COVID: {len(X_all)}/{len(X_train)}")
    print(f"retained COVID-input-overlap windows: {covid_overlap}")
    print(f"initial/final/minimum loss: {history[0]:.8f}/{history[-1]:.8f}/{min(history):.8f}")
    print(f"no-COVID growth metrics: {growth_metrics}")
    print(f"no-COVID level metrics: {level_metrics}")
    print(f"transformer tail quantiles: {_quantiles(transformer_growth)}")
    print(f"tail counts <-.01/-.02/-.05: {(transformer_growth < -.01).sum()}/{(transformer_growth < -.02).sum()}/{(transformer_growth < -.05).sum()}")
    print(f"worst-error concentration top1/top5/top10/top20: {sse[twenty[:1]].sum()/sse.sum():.4f}/{sse[twenty[:5]].sum()/sse.sum():.4f}/{sse[twenty[:10]].sum()/sse.sum():.4f}/{sse[twenty[:20]].sum()/sse.sum():.4f}")
    print(f"2024 transformer RMSE: {np.sqrt(sse_2024.mean()):.8f}")
    print(f"summary: {SUMMARY_PATH}")
    print(f"state metrics: {STATE_METRICS_PATH}")
    print(f"diagnostics: {DIAGNOSTICS_PATH}")


if __name__ == "__main__":
    main()
