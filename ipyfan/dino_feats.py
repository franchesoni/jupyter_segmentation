import math
import tqdm
import numpy as np
import torch
import skimage.transform

device = "cuda" if torch.cuda.is_available() else "cpu"
vitb8 = torch.hub.load("facebookresearch/dino:main", "dino_vitb8").to(device)


def image_as_input(img, device="cpu"):
    x = torch.tensor(img)
    x = x.unsqueeze(dim=0)  # make it a batch-like
    x = torch.einsum("nhwc->nchw", x)
    return x.float().to(device)


def dense_input(x, patch_side=8):
    H, W = x.shape[-2:]  # assume (..., H, W)
    x = torch.nn.functional.pad(
        x,
        (patch_side - 1, patch_side - 1, patch_side - 1, patch_side - 1),
        mode="reflect",
    )
    H_padded, W_padded = x.shape[-2:]  # assume (..., H_padded, W_padded)
    dense = []
    # with tqdm.tqdm(total=15**2) as pbar:
    for i in range((patch_side - 1) * 2 + 1):
        for j in range((patch_side - 1) * 2 + 1):
            # pbar.update(1)
            dense.append(x[..., i : i + 224, j : j + 224])
            # original image is i=7, j=7
    return torch.cat(dense)


def forward_ViT(model, x, use_pos_embed=True):
    model.eval()
    with torch.no_grad():
        # embed patches
        x = model.patch_embed(x)

        if use_pos_embed:
            # add pos embed w/o cls token
            x = x + model.pos_embed[:, 1:, :]

        # append cls token
        cls_token = model.cls_token + model.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in model.blocks:
            x = blk(x)
        x = model.norm(x)

    return x[:, 0], x[:, 1:]


def rolling_nanmean(model, x, patch_side=8, batch_size=1):
    assert (
        x.shape[0] == 1
    )  # passing the shifted versions through the model is the real bottleneck
    model.eval()
    with torch.no_grad():
        x = dense_input(x, patch_side=patch_side)
        B, C, H, W = x.shape
        assert B % batch_size == 0
        F, _, _ = 768, H // patch_side, W // patch_side
        out = torch.zeros((F, H, W), device=x.device)
        n = torch.zeros((H, W), device=x.device)
        if batch_size is not None:
            # for n_b in range(math.ceil(B / batch_size)):
            for n_b in tqdm.tqdm(range(math.ceil(B / batch_size))):

                x_in = x[batch_size * n_b : batch_size * (n_b + 1)]
                cls_feats, feats_b = forward_ViT(
                    model, x_in, use_pos_embed=True
                )
                del cls_feats
                feats_b = torch.einsum(
                    "nhwc->nchw", feats_b.reshape(batch_size, 28, 28, 768)
                )
                feats_b = torch.nn.functional.interpolate(
                    feats_b, scale_factor=8, mode="nearest"
                )  # (S, F, H, W)  # S is the number of shifted images
                # this array (feats_b) is HUGE
                s_i, s_j, e_i, e_j = (
                    (n_b * batch_size) // 15,
                    (n_b * batch_size) % 15,
                    ((n_b + 1) * batch_size) // 15,
                    ((n_b + 1) * batch_size) % 15,
                )
                e_ip, e_jp = (
                    (e_i, e_j - 1) if e_j != 0 else (e_i - 1, 14)
                )  # inclusive end
                # print(f"{s_i} <= i < {e_i}, {s_j} <= j < {e_j}")
                if s_i == e_ip:  # same row
                    i = s_i
                    for j in range(s_j, e_jp + 1):
                        # print(f"{i}, {j}, {i*15+j}/{(n_b+1)*batch_size-1}")
                        feats_b[
                            i * 15 + j - n_b * batch_size
                        ] = torch.nn.functional.pad(
                            feats_b[i * 15 + j - n_b * batch_size],
                            (j, 14 - j, i, 14 - i),
                            value=float("nan"),
                        )[
                            ..., 7 : 7 + 224, 7 : 7 + 224
                        ]
                elif s_i < e_ip:
                    i = s_i
                    for j in range(s_j, 15):
                        # print(f"{i}, {j}, {i*15+j}/{(n_b+1)*batch_size-1}")
                        feats_b[
                            i * 15 + j - n_b * batch_size
                        ] = torch.nn.functional.pad(
                            feats_b[i * 15 + j - n_b * batch_size],
                            (j, 14 - j, i, 14 - i),
                            value=float("nan"),
                        )[
                            ..., 7 : 7 + 224, 7 : 7 + 224
                        ]
                    for i in range(s_i + 1, e_ip):
                        for j in range(15):
                            feats_b[
                                i * 15 + j - n_b * batch_size
                            ] = torch.nn.functional.pad(
                                feats_b[i * 15 + j - n_b * batch_size],
                                (j, 14 - j, i, 14 - i),
                                value=float("nan"),
                            )[
                                ..., 7 : 7 + 224, 7 : 7 + 224
                            ]
                    i = e_ip
                    for j in range(e_jp + 1):
                        # print(f"{i}, {j}, {i*15+j}/{(n_b+1)*batch_size-1}")
                        feats_b[
                            i * 15 + j - n_b * batch_size
                        ] = torch.nn.functional.pad(
                            feats_b[i * 15 + j - n_b * batch_size],
                            (j, 14 - j, i, 14 - i),
                            value=float("nan"),
                        )[
                            ..., 7 : 7 + 224, 7 : 7 + 224
                        ]
                else:
                    raise RuntimeError("This should not happen")
                out = out + torch.nansum(feats_b, dim=0)
                n = n + torch.sum(
                    ~torch.isnan(feats_b[:, 0]), dim=0
                )  # take first feat
                del feats_b
            out = out / n[None]
        else:
            cls_feats, feats = forward_ViT(model, x, use_pos_embed=True)
            out = torch.einsum(
                "nhwc->nchw", feats.reshape(x.shape[0], 28, 28, -1)
            )
    return out


def process_img(img, batch_size=45):
    x = image_as_input(img, device)
    feat = rolling_nanmean(vitb8, x, batch_size=batch_size).cpu().numpy()
    return feat


def l2_channel_norm(x, channel_dim=-1):
    return x / np.expand_dims(
        np.linalg.norm(x, axis=channel_dim), axis=channel_dim
    )


def prepare_img(img):
    img = skimage.transform.resize(img, (224, 224), order=0)
    img = np.array(img) / 255.0
    img = img[..., :3]

    # normalize by ImageNet mean and std
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])
    img = img - imagenet_mean
    img = img / imagenet_std
    return img


def get_dino_feats(img, batch_size=1):
    out = prepare_img(img)
    out = process_img(out, batch_size=batch_size)
    out = l2_channel_norm(out, channel_dim=0)
    out = skimage.transform.resize(
        out.transpose(1, 2, 0), img.shape[:2], order=0
    )
    return out
