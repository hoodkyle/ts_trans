from torch import nn


def make_loss(name: str):
    if name == "mse":
        return nn.MSELoss()

    if name == "mae":
        return nn.L1Loss()

    if name == "huber":
        return nn.HuberLoss()

    raise ValueError(f"unknown loss: {name}")