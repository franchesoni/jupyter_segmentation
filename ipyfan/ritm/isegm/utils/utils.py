from pathlib import Path

import torch

from .serialization import load_model


def load_is_model(checkpoint, device, **kwargs):
    if isinstance(checkpoint, (str, Path)):
        state_dict = torch.load(checkpoint, map_location="cpu")
    else:
        state_dict = checkpoint

    if isinstance(state_dict, list):
        model = load_single_is_model(state_dict[0], device, **kwargs)
        models = [
            load_single_is_model(x, device, **kwargs) for x in state_dict
        ]

        return model, models
    else:
        # (removed) hack it to iis_framework
        state_dict["config"]["class"] = (
            state_dict["config"]["class"]
            # "models.custom.ritm." + state_dict["config"]["class"]
        )
        return load_single_is_model(state_dict, device, **kwargs)


def load_single_is_model(state_dict, device, **kwargs):
    model = load_model(state_dict["config"], **kwargs)
    model.load_state_dict(state_dict["state_dict"], strict=False)

    for param in model.parameters():
        param.requires_grad = False
    model.to(device)
    model.eval()

    return model
