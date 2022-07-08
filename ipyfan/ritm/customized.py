from pathlib import Path
from typing import Union
import numpy as np

import torch
from .isegm.utils.utils import load_is_model


def norm_fn(
    x: Union[torch.Tensor, np.ndarray]
) -> Union[torch.Tensor, np.ndarray]:
    if (xmax := x.max()) != (xmin := x.min()):
        return (x - xmin) / (xmax - xmin)
    if isinstance(x, np.ndarray):
        return np.ones_like(x) * xmax
    if isinstance(x, torch.Tensor):
        return torch.ones_like(x) * xmax
    raise ValueError(f"`x` type is {type(x)} instead of Tensor or ndarray")


def predict(model, image, prev_mask, pcs, ncs):
    assert len(pcs[0]) == len(ncs[0]), "batch size should be consistent"
    assert len(pcs) == len(ncs), "interactions should be consistent"
    for n_interaction in range(len(ncs)):
        for n_element in range(len(ncs[0])):
            maxlen = max(
                len(pcs[n_interaction][n_element]),
                len(ncs[n_interaction][n_element]),
            )
            pcs[n_interaction][n_element] = pcs[n_interaction][n_element] + [
                -1
            ] * (maxlen - len(pcs[n_interaction][n_element]))
            ncs[n_interaction][n_element] = ncs[n_interaction][n_element] + [
                -1
            ] * (maxlen - len(ncs[n_interaction][n_element]))

    points = []
    for n_element in range(len(pcs[0])):  # for element in batch
        points.append([])
        for n_interaction in range(len(pcs)):  # for each interaction
            for click in pcs[n_interaction][n_element]:
                if (click == -1) is not True:
                    points[-1].append(
                        [click[0], click[1], n_interaction]
                    )  # add all positive clicks
                else:
                    points[-1].append([-1, -1, -1])  # add all positive clicks

        for n_interaction in range(len(ncs)):
            for click in ncs[n_interaction][n_element]:
                if (click == -1) is not True:
                    points[-1].append(
                        [click[0], click[1], n_interaction]
                    )  # add all negative clicks
                else:
                    points[-1].append([-1, -1, -1])  # add all positive clicks

    input_image = torch.cat((image, prev_mask), dim=1)
    pred_logits = model.forward(input_image, torch.Tensor(points))["instances"]
    prediction = torch.nn.functional.interpolate(
        pred_logits,
        mode="bilinear",
        align_corners=True,
        size=input_image.size()[2:],
    )
    prediction = torch.sigmoid(prediction)
    return prediction, {"prev_prediction": prediction}


def initialize_z(image, target):
    return {"prev_prediction": torch.zeros_like(image[:, :1, :, :])}


def initialize_y(image, target):
    y = torch.zeros_like(target)
    return y


CHECKPOINT = str(Path(__file__).parent / "coco_lvis_h18s_itermask.pth")
model = load_is_model(CHECKPOINT, device="cpu")


def ritm(x, z, pcs, ncs, checkpoint):
    assert (
        CHECKPOINT == checkpoint
    ), f"checkpoint should be consistent but is {CHECKPOINT} and {checkpoint}"
    x = norm_fn(x)
    y, z = predict(model, x, z["prev_prediction"], pcs, ncs)
    return y, z
