"""Run a mini-batch scaled training experiment on sinusoidal panels."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from ts_trans.dataprep import inverse_standardize, make_panel_windows, standardize_panel
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


def _metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = prediction - target
    return {
        "mse": float(np.mean(error**2)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
    }


def main() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    root = Path(__file__).resolve().parents[1]
    data = pd.read_csv(root / "data" / "testdata" / "panels" / "sinusoidal_panel.csv")
    parameters = pd.read_csv(root / "data" / "testdata" / "panels" / "sinusoidal_parameters.csv")
    scaled_data, scaling_params = standardize_panel(data, ["panel"], "value")
    X, y, metadata = make_panel_windows(
        scaled_data, ["panel"], "date", "value", "monthly", WINDOW_LENGTH
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)
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
            predictions = model(batch_X)
            loss = loss_fn(predictions, batch_y)
            loss.backward()
            optimizer.step()
            batch_size = batch_X.shape[0]
            total_loss += loss.item() * batch_size
            total_items += batch_size
        epoch_loss = total_loss / total_items
        epoch_history.append(epoch_loss)
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1:
            print(f"epoch {epoch + 1:3d}/{EPOCHS}: loss={epoch_loss:.6f}")

    model.eval()
    with torch.no_grad():
        standardized_predictions = model(X_tensor.to(device)).cpu().numpy()
    standardized_targets = y
    original_predictions = inverse_standardize(
        standardized_predictions, metadata, scaling_params, ["panel"]
    )
    original_targets = inverse_standardize(standardized_targets, metadata, scaling_params, ["panel"])
    last_value_predictions = inverse_standardize(X[:, -1, :], metadata, scaling_params, ["panel"])
    lag12_predictions = inverse_standardize(X[:, 0, :], metadata, scaling_params, ["panel"])

    standardized_metrics = _metrics(standardized_targets, standardized_predictions)
    original_metrics = _metrics(original_targets, original_predictions)
    last_value_metrics = _metrics(original_targets, last_value_predictions)
    lag12_metrics = _metrics(original_targets, lag12_predictions)

    enriched_metadata = metadata.merge(parameters, on="panel", how="left", validate="many_to_one")
    bands = pd.cut(
        enriched_metadata["wavelength"],
        bins=[12, 24, 48, 72, 96],
        labels=["12-24 months", "24-48 months", "48-72 months", "72-96 months"],
        include_lowest=True,
    )
    enriched_metadata["wavelength_band"] = bands
    band_lines = []
    for band, group in enriched_metadata.groupby("wavelength_band", observed=False):
        indices = group.index.to_numpy()
        band_metrics = _metrics(original_targets[indices], original_predictions[indices])
        band_lines.append(
            f"  {band}: windows={len(indices)}, rmse={band_metrics['rmse']:.10f}, mae={band_metrics['mae']:.10f}"
        )

    summary_lines = [
        "Scaled sinusoidal panel training summary",
        f"device: {device}",
        f"windows: {len(X)}",
        f"batch_size: {BATCH_SIZE}",
        f"batches_per_epoch: {len(loader)}",
        f"learned_parameters: {parameter_count}",
        f"window_length: {WINDOW_LENGTH}",
        f"d_model: {D_MODEL}",
        f"d_ff: {D_FF}",
        f"n_blocks: {N_BLOCKS}",
        f"learning_rate: {LEARNING_RATE}",
        f"epochs: {EPOCHS}",
        f"seed: {SEED}",
        f"loss: {LOSS}",
        f"initial_epoch_loss: {epoch_history[0]:.10f}",
        f"final_epoch_loss: {epoch_history[-1]:.10f}",
        f"minimum_epoch_loss: {min(epoch_history):.10f}",
        "standardized_metrics:",
        *(f"  {name}: {value:.10f}" for name, value in standardized_metrics.items()),
        "original_metrics:",
        *(f"  {name}: {value:.10f}" for name, value in original_metrics.items()),
        "last_value_metrics:",
        *(f"  {name}: {value:.10f}" for name, value in last_value_metrics.items()),
        "lag12_metrics:",
        *(f"  {name}: {value:.10f}" for name, value in lag12_metrics.items()),
        "wavelength_bands:",
        *band_lines,
        "sample_predictions:",
    ]
    for index in range(min(5, len(metadata))):
        row = enriched_metadata.iloc[index]
        sample = {
            "panel": row["panel"],
            "forecast_time": row["forecast_time"],
            "amplitude": float(row["amplitude"]),
            "wavelength": float(row["wavelength"]),
            "target_original": float(original_targets[index, 0]),
            "prediction_original": float(original_predictions[index, 0]),
            "error_original": float(original_predictions[index, 0] - original_targets[index, 0]),
            "last_value_prediction": float(last_value_predictions[index, 0]),
            "lag12_prediction": float(lag12_predictions[index, 0]),
        }
        summary_lines.append(str(sample))

    output_path = root / "output" / "train_sinusoidal_scaled_summary.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(summary_lines) + "\n")
    print(f"device: {device}")
    print(f"windows: {len(X)}")
    print(f"batch size: {BATCH_SIZE}; batches per epoch: {len(loader)}")
    print(f"learned parameters: {parameter_count}")
    print(f"initial/final/minimum epoch loss: {epoch_history[0]:.6f}/{epoch_history[-1]:.6f}/{min(epoch_history):.6f}")
    print(f"standardized metrics: {standardized_metrics}")
    print(f"original metrics: {original_metrics}")
    print(f"last-value metrics: {last_value_metrics}")
    print(f"lag-12 metrics: {lag12_metrics}")
    print("wavelength bands:")
    print("\n".join(band_lines))
    for index in range(min(5, len(metadata))):
        row = enriched_metadata.iloc[index]
        print(
            {
                "panel": row["panel"],
                "forecast_time": row["forecast_time"],
                "amplitude": float(row["amplitude"]),
                "wavelength": float(row["wavelength"]),
                "target_original": float(original_targets[index, 0]),
                "prediction_original": float(original_predictions[index, 0]),
                "error_original": float(original_predictions[index, 0] - original_targets[index, 0]),
                "last_value_prediction": float(last_value_predictions[index, 0]),
                "lag12_prediction": float(lag12_predictions[index, 0]),
            }
        )
    print(f"summary: {output_path}")


if __name__ == "__main__":
    main()
