import torch
import math
import numpy as np

from ipyfan.utils import rgba2rgb, norm_fn


import sys

sys.path.append("../../iislib/iislib")
# sys.path.append('../iislib')
def get_model_ritm(checkpoint_path):
    from models.custom.ritm.customized import initialize_y, initialize_z, ritm
    import functools

    ritm_fn = functools.partial(ritm, checkpoint=checkpoint_path)
    return ritm_fn, initialize_z, initialize_y


checkpoint = "../../iislib/uneteffb0/checkpoints/last_checkpoint.pth"  # unet effnet b0
# checkpoint = "/home/franchesoni/iis/ritm_interactive_segmentation/experiments/iter_mask/sbd_unet/012_unet_effb0/checkpoints/last_checkpoint.pth"  # unet tiny
ritm_model, init_z, init_y = get_model_ritm(checkpoint)

if hasattr(ritm_model, "eval"):
    ritm_model.eval()


def adapt_inputs_demo(curr_im, curr_ref, list_of_pcs, list_of_ncs):
    assert (
        len(curr_im.shape) == 2
    ), f"only grayscale image suported but received {curr_im.shape}"
    assert curr_ref.shape[2] == 4, f"should be rgba image but is {curr_ref.shape}"
    img = torch.Tensor(np.repeat(norm_fn(curr_im[None, ...]), 3, axis=0))[
        None, ...
    ]  # channel first
    normref = norm_fn(curr_ref.sum(axis=2))
    z = {"prev_prediction": torch.Tensor(normref)[None, None, ...]}
    pcs = [[[click]] for click in list_of_pcs] or [
        [[]]
    ]  # interaction, batch, click_number, (x,y)
    ncs = [[[click]] for click in list_of_ncs] or [[[]]]
    return img, z, pcs, ncs


def adapt_inputs(curr_im, curr_ref, list_of_pcs, list_of_ncs):
    assert (
        curr_im.shape[2] == 4
    ), f"only rgba image suported but received {curr_im.shape}"
    assert curr_ref.shape[2] == 4, f"should be rgba image but is {curr_ref.shape}"
    img = (
        torch.Tensor(norm_fn(rgba2rgb(curr_im)) * 255)
        .permute(2, 0, 1)[None, ...]
        .float()
    )  # channel first
    normref = norm_fn(curr_ref.sum(axis=2)) * 255
    normref = torch.Tensor(normref)[None, None, ...].float()
    assert (
        img.shape[-1] == img.shape[-2] == normref.shape[-1] == normref.shape[-2]
    ), "image should be square"
    target_shape = math.ceil(img.shape[-1] / 32) * 32
    padl = (target_shape - img.shape[-1]) // 2
    padr = (target_shape - img.shape[-1]) // 2 + (target_shape - img.shape[-1]) % 2
    img = torch.nn.functional.pad(img, (padl, padr, padl, padr))
    normref = torch.nn.functional.pad(normref, (padl, padr, padl, padr))
    print(img.shape, normref.shape)
    z = {"prev_prediction": normref}

    pcs = [[[[padl + click[0], padl + click[1]]]] for click in list_of_pcs] or [
        [[]]
    ]  # interaction, batch, click_number, (x,y)
    ncs = [[[[padl + click[0], padl + click[1]]]] for click in list_of_ncs] or [[[]]]
    return img, z, pcs, ncs, padl, padr


def adapt_outputs(y, z, padl, padr):
    if 0 < padr:
        return y[0, 0, padl:-padr, padl:-padr].detach().numpy()
    else:
        return y[0, 0, padl:, padl:].detach().numpy()


def iis_model(img, ref, list_of_pcs, list_of_ncs):
    img, z, pcs, ncs, padl, padr = adapt_inputs(img, ref, list_of_pcs, list_of_ncs)
    y, z = ritm_model(img, z, pcs, ncs)
    return adapt_outputs(y, z, padl, padr)


def iis_model_demo(img, ref, list_of_pcs, list_of_ncs):
    y, z = ritm_model(*adapt_inputs_demo(img, ref, list_of_pcs, list_of_ncs))
    return adapt_outputs(y, z, 0, 0)


with torch.no_grad():
    dummy_input = [
        (255 * np.random.rand(500, 500, 4)).astype(np.uint8),
        np.zeros((500, 500, 4), dtype="uint8"),
        [[237, 102]],
        [],
    ]
    out = iis_model(*dummy_input)

print(out.shape, out.max(), out.min(), out.dtype, type(out))
