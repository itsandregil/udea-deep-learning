import os

import torch


def save_model(model: torch.nn.Module, filename: str, *, dirname="models"):
    if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)

    _, extension = filename.split(".", maxsplit=1)
    if extension != "pth":
        raise ValueError(f"File extension {extension} is not supported, change to .pth")

    try:
        torch.save(model.state_dict(), os.path.join(dirname, filename))
    except Exception as e:
        raise Exception(f"Error found while trying to save the model: {e}")
