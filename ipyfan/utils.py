"""
This entire file was copied from ipycanvas by Martin Renou
https://github.com/martinRenou/ipycanvas/blob/4d1fdea1b42bd963115cb881ee9d0810ef2a9913/ipycanvas/utils.py
"""

"""Binary module."""
from io import BytesIO

# from PIL import Image as PILImage

import numpy as np

def str_description(x):
    return f"{x.shape, x.min(), x.max(), x.dtype}"

# def image_bytes_to_array(im_bytes):
#     """Turn raw image bytes into a NumPy array."""
#     im_file = BytesIO(im_bytes)

#     im = PILImage.open(im_file)

#     return np.array(im)
def rgba2rgb(rgba, background=(255, 255, 255)):
    row, col, ch = rgba.shape
    if ch == 3:
        return rgba
    assert ch == 4, "RGBA image has 4 channels."
    rgb = np.zeros((row, col, 3), dtype="float32")
    r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]
    a = np.asarray(a, dtype="float32") / 255.0
    R, G, B = background
    rgb[:, :, 0] = r * a + (1.0 - a) * R
    rgb[:, :, 1] = g * a + (1.0 - a) * G
    rgb[:, :, 2] = b * a + (1.0 - a) * B
    return np.asarray(rgb, dtype="uint8")


def unpad(img, padl, padr, padt, padb, channels_last=True):
    if channels_last:
        if 0 < padr:
            out = img[:, padl:-padr]
        else:
            out = img[:, padl:]
        if 0 < padt:
            out = out[padb:-padt]
        else:
            out = out[padb:]
    else:
        if 0 < padr:
            out = img[..., :, padl:-padr]
        else:
            out = img[..., :, padl:]
        if 0 < padt:
            out = out[..., padb:-padt]
        else:
            out = out[..., padb:]
    return out


# def unpad(img, padl, padr, channels_last=True):
#     if channels_last:
#         if 0 < padr:
#             return img[padl:-padr, padl:-padr]
#         else:
#             return img[padl:, padl:]
#     else:
#         if 0 < padr:
#             return img[..., padl:-padr, padl:-padr]
#         else:
#             return img[..., padl:, padl:]


def to_np(img):
    if 3 <= len(img.shape):
        out = img
        while 3 < len(out.shape):
            out = out[0]
    else:
        out = img[None]
    out = (
        out.permute(1, 2, 0) if out.shape[0] < 4 else out
    )  # assume images bigger than 5x5
    out = np.array(out)
    out = out / 255 if 1 < out.max() else out
    return out


def norm_fn(x):
    xmin, xmax = x.min(), x.max()
    if xmin == xmax:
        if xmax == 0:
            return np.zeros_like(x, dtype="float32")
        return x
    else:
        return (x - xmin) / (xmax - xmin)


def binary_image(ar):
    """Turn a NumPy array representing an array of pixels into a binary buffer."""
    if ar is None:
        return None
    if ar.dtype != np.uint8:
        ar = ar.astype(np.uint8)
    if ar.ndim == 1:
        ar = ar[np.newaxis, :]
    if ar.ndim == 2:
        # extend grayscale to RGBA
        add_alpha = np.full((ar.shape[0], ar.shape[1], 4), 255, dtype=np.uint8)
        add_alpha[:, :, :3] = np.repeat(ar[:, :, np.newaxis], repeats=3, axis=2)
        ar = add_alpha
    if ar.ndim != 3:
        raise ValueError("Please supply an RGBA array with shape (width, height, 4).")
    if ar.shape[2] != 4 and ar.shape[2] == 3:
        add_alpha = np.full((ar.shape[0], ar.shape[1], 4), 255, dtype=np.uint8)
        add_alpha[:, :, :3] = ar
        ar = add_alpha
    if not ar.flags["C_CONTIGUOUS"]:  # make sure it's contiguous
        ar = np.ascontiguousarray(ar, dtype=np.uint8)
    return {"shape": ar.shape, "dtype": str(ar.dtype)}, memoryview(ar)


def array_to_binary(ar):
    """Turn a NumPy array into a binary buffer."""
    # Unsupported int64 array JavaScript side
    if ar.dtype == np.int64:
        ar = ar.astype(np.int32)

    # Unsupported float16 array JavaScript side
    if ar.dtype == np.float16:
        ar = ar.astype(np.float32)

    # make sure it's contiguous
    if not ar.flags["C_CONTIGUOUS"]:
        ar = np.ascontiguousarray(ar)

    return {"shape": ar.shape, "dtype": str(ar.dtype)}, memoryview(ar)


def populate_args(arg, args, buffers):
    if isinstance(arg, (list, np.ndarray)):
        arg_metadata, arg_buffer = array_to_binary(np.asarray(arg))
        arg_metadata["idx"] = len(buffers)

        args.append(arg_metadata)
        buffers.append(arg_buffer)
    else:
        args.append(arg)


def to_camel_case(snake_str):
    """Snake case to Camel case translator."""
    components = snake_str.split("_")
    return components[0] + "".join(x.title() for x in components[1:])

def make_psd(sigma):
    C = (sigma + sigma.T) / 2
    eigval, eigvec = np.linalg.eigh(C)
    eigval = eigval * (0 < eigval)
    sigma_psm = eigvec @ np.diag(eigval) @ eigvec.T
    return sigma_psm