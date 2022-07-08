import torch
import numpy as np
import math
from .utils import norm_fn, rgba2rgb, to_np, unpad

from pathlib import Path

# sys.path.append(str(Path(__file__).parent.parent.parent / "ritm"))
from .ritm.visualization import visualize_on_test

# sys.path.append(str(Path(__file__).parent.parent.parent / 'iislib/tests'))


def get_model_ritm(checkpoint_path):
    from .ritm.customized import initialize_y, initialize_z, ritm

    import functools

    ritm_fn = functools.partial(ritm, checkpoint=checkpoint_path)
    return ritm_fn, initialize_z, initialize_y


checkpoint = str(
    Path(__file__).parent / "ritm/coco_lvis_h18s_itermask.pth"
)
ritm_model, init_z, init_y = get_model_ritm(checkpoint)

if hasattr(ritm_model, "eval"):
    ritm_model.eval()


def adapt_inputs(curr_ref, curr_im, z, list_of_pcs, list_of_ncs, logger=None):
    assert (
        curr_im.shape[2] == 4
    ), f"only rgba image suported but received {curr_im.shape}"
    assert (
        curr_ref.shape[2] == 4
    ), f"should be rgba image but is {curr_ref.shape}"
    img = (
        torch.Tensor(norm_fn(rgba2rgb(curr_im)) * 255)
        .permute(2, 0, 1)[None, ...]
        .float()
    )  # channel first
    normref = norm_fn(curr_ref.sum(axis=2)) * 255
    normref = torch.Tensor(normref)[None, None, ...].float()
    assert (
        img.shape[-1]
        == img.shape[-2]
        == normref.shape[-1]
        == normref.shape[-2]
    ), "image should be square"
    target_shape = math.ceil(img.shape[-1] / 32) * 32
    padl = (target_shape - img.shape[-1]) // 2
    padr = (target_shape - img.shape[-1]) // 2 + (
        target_shape - img.shape[-1]
    ) % 2
    img = torch.nn.functional.pad(img, (padl, padr, padl, padr))
    normref = torch.nn.functional.pad(normref, (padl, padr, padl, padr))
    z = {
        "prev_prediction": normref,
        "prev_output": normref,
    }  # I used different names :/

    pcs = [
        [[[padl + click[1], padl + click[0]]]] for click in list_of_pcs
    ] or [
        [[]]
    ]  # interaction, batch, click_number, (x,y)
    ncs = [
        [[[padl + click[1], padl + click[0]]]] for click in list_of_ncs
    ] or [[[]]]
    maxlen = max(len(pcs), len(ncs))
    pcs = pcs + [[[]]] * (maxlen - len(pcs))
    ncs = ncs + [[[]]] * (maxlen - len(ncs))
    if logger:
        logger.info(f"pcs = {pcs}, ncs = {ncs}")
    return img, z, pcs, ncs, padl, padr


def adapt_outputs(y, z, padl, padr, logger=None):
    if "prev_output" in z:
        z["prev_prediction"] = z["prev_output"]
    elif "prev_prediction" in z:
        z["prev_output"] = z["prev_prediction"]
    return unpad(y[0][0], padl, padr).detach().numpy()


def wrapped_model(ref, img, z, list_of_pcs, list_of_ncs, logger):
    img_in, z, pcs, ncs, padl, padr = adapt_inputs(
        ref, img, z, list_of_pcs, list_of_ncs, logger
    )
    # (1, 3, H, W), {
    # 'prev_output/prev_prediction': (1, 1, H, W)}, [[[(x,y)]]], [[[(x,y)]]]
    y, new_z = ritm_model(img_in, z, pcs, ncs)
    y, new_z = adapt_outputs(y, new_z, padl, padr, logger)
    if logger:
        # log shapes
        aa = to_np(unpad(img_in[0], padl, padr, channels_last=False)).shape
        bb = (
            to_np(
                unpad(z["prev_prediction"][0], padl, padr, channels_last=False)
            ).squeeze()
        ).shape
        logger.info(f"{aa}, {bb}, {y.shape}")
        visualize_on_test(
            to_np(unpad(img_in[0], padl, padr, channels_last=False)),
            to_np(
                unpad(z["prev_prediction"][0], padl, padr, channels_last=False)
            ).squeeze(),  # channels last
            y,
            pcs,
            ncs,
            name="model_output",
            destdir=".",
        )
    return y, new_z


if __name__ == "__main__":
    with torch.no_grad():
        dummy_input = [
            (255 * np.random.rand(500, 500, 4)).astype(np.uint8),
            np.zeros((500, 500, 4), dtype="uint8"),
            [[237, 102]],
            [],
        ]
        out = wrapped_model(*dummy_input, logger=None)
