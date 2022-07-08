import os
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch


def visualize(image, target, output, pcs, ncs, name, alpha=0.3):
    assert image.shape[2] == 3 and len(image.shape) == 3
    assert len(target.shape) == 2 == len(output.shape)

    image = norm_fn(np.array(image))
    mean_color = image.sum((0, 1)) / image.size
    min_rgb = np.argmin(mean_color)  # min of (mean R, mean G, mean B)
    mask_color = np.ones((1, 1, 3)) * np.eye(3)[min_rgb][None, None, :]
    out = (
        image * (1 - alpha)
        + (1 * (0.5 < norm_fn(output)))[:, :, None] * mask_color * alpha
    )  # add inverted mean-color mask

    plt.figure(figsize=(4, 4))

    plt.subplot(221)
    plt.imshow(image)
    plt.grid()
    plt.axis("off")

    plt.subplot(222)
    plt.imshow(out)
    for cind, pc_list in enumerate(pcs):
        for pc in pc_list[0]:  # assume batch=1
            if (pc == -1) is not True:
                plt.scatter(pc[1], pc[0], s=40 // len(pcs) * (cind + 1), color="g")
    for cind, nc_list in enumerate(ncs):
        for nc in nc_list[0]:  # assume batch=1
            if (nc == -1) is not True:
                plt.scatter(nc[1], nc[0], s=40 // len(pcs) * (cind + 1), color="r")
    plt.grid()
    plt.axis("off")

    plt.subplot(223)
    plt.imshow(
        np.stack(
            (norm_fn(target), 1 * (0.5 < norm_fn(output)), norm_fn(output)),
            axis=2,
        )
    )
    for cind, pc_list in enumerate(pcs):
        for pc in pc_list[0]:  # assume batch=1
            if (pc == -1) is not True:
                plt.scatter(pc[1], pc[0], s=40 // len(pcs) * (cind + 1), color="g")
    for cind, nc_list in enumerate(ncs):
        for nc in nc_list[0]:  # assume batch=1
            if (nc == -1) is not True:
                plt.scatter(nc[1], nc[0], s=40 // len(pcs) * (cind + 1), color="r")
    plt.grid()
    plt.axis("off")

    plt.subplot(224)
    plt.imshow(norm_fn(output))
    for cind, pc_list in enumerate(pcs):
        for pc in pc_list[0]:  # assume batch=1
            if (pc == -1) is not True:
                plt.scatter(pc[1], pc[0], s=40 // len(pcs) * (cind + 1), color="g")
    for cind, nc_list in enumerate(ncs):
        for nc in nc_list[0]:  # assume batch=1
            if (nc == -1) is not True:
                plt.scatter(nc[1], nc[0], s=40 // len(pcs) * (cind + 1), color="r")
    plt.grid()
    plt.axis("off")

    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    fig = plt.gcf()
    return fig


def visualize_on_test(image, target, output, pcs, ncs, name, alpha=0.3, destdir=None):
    fig = visualize(image, target, output, pcs, ncs, name, alpha)

    if destdir is None:
        fig.savefig(name + ".png")
    else:
        Path(destdir).mkdir(exist_ok=True)
        fig.savefig(os.path.join(destdir, name + ".png"))
    plt.close()


def visualize_clicks(image, mask, alpha, pc_list, nc_list, name):

    image = norm_fn(np.array(image))
    mean_color = image.sum((0, 1)) / image.size
    min_rgb = np.argmin(mean_color)
    mask_color = np.ones((1, 1, 3)) * np.eye(3)[min_rgb][None, None, :]
    out = (
        image / image.max() * (1 - alpha) + mask[:, :, None] * mask_color * alpha
    )  # add inverted mean color mask
    plt.figure()
    plt.imshow(out)
    plt.axis("off")
    plt.grid()
    for pc in pc_list:
        plt.scatter(pc[1], pc[0], s=10, color="g")
    for nc in nc_list:
        plt.scatter(nc[1], nc[0], s=10, color="r")
    plt.savefig(name + ".png")
    plt.close()


def save_as_png(img, name):
    """Save image as png"""
    plt.figure()
    plt.imshow(img)
    plt.axis("off")
    plt.grid()
    plt.savefig(name + ".png")
    plt.close()


def norm_fn(x: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    if (xmax := x.max()) != (xmin := x.min()):
        return (x - xmin) / (xmax - xmin)
    if isinstance(x, np.ndarray):
        return np.ones_like(x) * xmax
    if isinstance(x, torch.Tensor):
        return torch.ones_like(x) * xmax
    raise ValueError(f"`x` type is {type(x)} instead of Tensor or ndarray")
