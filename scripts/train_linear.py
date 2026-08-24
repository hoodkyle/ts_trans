"""Run one small full-batch smoke-training experiment on linear panel data."""

from pathlib import Path

import pandas as pd
import torch

from ts_trans.dataprep import make_panel_windows
from ts_trans.model.forecast import TimeSeriesTransformer
from ts_trans.training import make_loss, train_model


WINDOW_LENGTH = 12
D_MODEL = 8
D_FF = 16
N_BLOCKS = 2
LEARNING_RATE = 1e-3
EPOCHS = 20
SEED = 20260824


def main() -> None:
    torch.manual_seed(SEED)
    data_path = Path(__file__).resolve().parents[1] / "data" / "testdata" / "panels" / "linear_panel.csv"
    data = pd.read_csv(data_path)
    X, y, _ = make_panel_windows(data, ["panel"], "date", "value", "monthly", WINDOW_LENGTH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y, dtype=torch.float32, device=device)
    model = TimeSeriesTransformer(D_MODEL, D_FF, N_BLOCKS).to(device)
    loss_fn = make_loss("mse")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    model.train()
    with torch.no_grad():
        initial_loss = loss_fn(model(X_tensor), y_tensor).item()
    history = train_model(model, X_tensor, y_tensor, loss_fn, optimizer, EPOCHS)

    print(f"device: {device}")
    print(f"X.shape: {X.shape}")
    print(f"y.shape: {y.shape}")
    print(f"learned parameters: {sum(parameter.numel() for parameter in model.parameters())}")
    print(f"initial loss: {initial_loss:.6f}")
    print(f"final loss: {history[-1]:.6f}")
    print(f"first 5 losses: {[round(loss, 6) for loss in history[:5]]}")
    print(f"last 5 losses: {[round(loss, 6) for loss in history[-5:]]}")

    model.eval()
    with torch.no_grad():
        predictions = model(X_tensor[:3]).cpu().numpy().reshape(-1)
    for index, prediction in enumerate(predictions):
        print(f"sample {index}: target={y[index, 0]:.6f}, prediction={prediction:.6f}, error={prediction - y[index, 0]:.6f}")


if __name__ == "__main__":
    main()
