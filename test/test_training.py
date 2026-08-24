import torch
from torch import nn
import pytest

from ts_trans.training import make_loss, train_model


@pytest.mark.parametrize(
    ("name", "expected_type"),
    [("mse", nn.MSELoss), ("mae", nn.L1Loss), ("huber", nn.HuberLoss)],
)
def test_loss_selector(name, expected_type):
    assert isinstance(make_loss(name), expected_type)


def test_loss_selector_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown loss"):
        make_loss("not-a-loss")


def test_train_model_updates_toy_model_and_reduces_loss():
    torch.manual_seed(20260824)
    model = nn.Linear(1, 1)
    X = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    y = 2.0 * X + 1.0
    loss_fn = make_loss("mse")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    initial_parameters = [parameter.detach().clone() for parameter in model.parameters()]
    initial_loss = loss_fn(model(X), y).item()

    history = train_model(model, X, y, loss_fn, optimizer, epochs=25)

    assert len(history) == 25
    assert all(torch.isfinite(torch.tensor(loss)) for loss in history)
    assert history[-1] < initial_loss
    assert any(not torch.equal(before, after) for before, after in zip(initial_parameters, model.parameters()))
