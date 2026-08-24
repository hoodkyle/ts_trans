import torch


def train_model(
    model,
    X,
    y,
    loss_fn,
    optimizer,
    epochs: int,
):
    model.train()

    history = []

    for epoch in range(epochs):
        optimizer.zero_grad()

        predictions = model(X)
        loss = loss_fn(predictions, y)

        loss.backward()
        optimizer.step()

        history.append(loss.item())

    return history