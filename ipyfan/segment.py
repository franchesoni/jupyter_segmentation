#!/usr/bin/env python
# coding: utf-8

# Copyright (c) franchesoni.
# Distributed under the terms of the Modified BSD License.

"""
TODO: Add module docstring
"""

from ipywidgets import DOMWidget, Output

from ._frontend import module_name, module_version
from .utils import (
    binary_image,
    norm_fn,
    unpad,
    rgba2rgb,
    to_np,
    make_psd,
    str_description,
)
from numpy import frombuffer, uint8, copy
import zlib
from ipydatawidgets import (
    shape_constraints,
    DataUnion,
    data_union_serialization,
)
import numpy as np

import sklearn.mixture
import sklearn.cluster
import skimage.segmentation
import skimage.transform
import scipy.signal

import math
import torch

from traitlets import Bytes, CInt, Unicode, Float, List
from traitlets import observe
import ipywidgets as widgets

import logging

from .ritm.visualization import visualize_on_test

logging.basicConfig(
    filename="backend.log", encoding="utf-8", level=logging.DEBUG, filemode="w"
)
logger = logging.getLogger("matplotlib").setLevel(logging.WARNING)
logger = logging.getLogger()


def deserializeImage(json, obj):
    logger.info("deserializing:", json)
    _bytes = None if json["data"] is None else json["data"].tobytes()
    if _bytes is not None:
        # print(sys.getsizeof(_bytes))
        _bytes = zlib.decompress(_bytes)
        # print(sys.getsizeof(_bytes))
        # copy to make it a writeable array - not necessary, but nice
        obj.labels = copy(
            frombuffer(_bytes, dtype=uint8).reshape(
                json["width"], json["height"], 4
            )
        )
    return _bytes


labels_serialization = {
    "from_json": deserializeImage,
}

debug_view = Output(layout={"border": "1px solid black"})


class segmenter(DOMWidget):
    """TODO: Add docstring here"""

    _model_name = Unicode("segmentModel").tag(sync=True)
    _model_module = Unicode(module_name).tag(sync=True)
    _model_module_version = Unicode(module_version).tag(sync=True)
    _view_name = Unicode("segmentView").tag(sync=True)
    _view_module = Unicode(module_name).tag(sync=True)
    _view_module_version = Unicode(module_version).tag(sync=True)

    tool = CInt(0).tag(sync=True)
    alpha = Float(0.3).tag(sync=True)
    size = CInt(10).tag(sync=True)
    pcs = List([]).tag(sync=True)
    ncs = List([]).tag(sync=True)
    s_x, s_y, s_width, s_height = (
        Float(0.0).tag(sync=True),
        Float(0.0).tag(sync=True),
        Float(0.0).tag(sync=True),
        Float(0.0).tag(sync=True),
    )

    # underlying info for labels - this handles the syncing to ts
    _labels = Bytes(default_value=None, allow_none=True, read_only=True).tag(
        sync=True, **labels_serialization
    )

    # proposal = DataUnion(
    propL = DataUnion(
        np.zeros([10, 10, 4], dtype=np.uint8),
        dtype="uint8",
        shape_constraint=shape_constraints(None, None, 4),  # 2D RGBA
    ).tag(sync=True, **data_union_serialization)

    prevPropL = DataUnion(  # preview to build proposal
        np.zeros([10, 10, 4], dtype=np.uint8),
        dtype="uint8",
        shape_constraint=shape_constraints(None, None, 4),  # 2D RGBA
    ).tag(sync=True, **data_union_serialization)

    # data = DataUnion(
    annI = DataUnion(
        np.zeros([10, 10, 4], dtype=np.uint8),
        dtype="uint8",
        shape_constraint=shape_constraints(None, None, 4),  # 2D RGBA
    ).tag(sync=True, **data_union_serialization)

    annL = DataUnion(
        np.zeros([10, 10, 4], dtype=np.uint8),
        dtype="uint8",
        shape_constraint=shape_constraints(None, None, 4),  # 2D RGBA
    ).tag(sync=True, **data_union_serialization)

    imgL = DataUnion(
        np.zeros([10, 10, 4], dtype=np.uint8),
        dtype="uint8",
        shape_constraint=shape_constraints(None, None, 4),  # 2D RGBA
    ).tag(sync=True, **data_union_serialization)

    logger.info("end class variables")

    def __init__(self, iis_model=None, **kwargs):
        logger.debug("init segmenter")
        super().__init__(**kwargs)
        self.iis_model = iis_model
        self.iis_state = {}
        self.superpix_state = {}
        self.cluster_state = {}

    # push: propL, prevPropL, annI
    # receive: annI, annL, imgL, tool, alpha, size, pcs, ncs
    @observe(
        "tool",
        "alpha",
        "size",
        "pcs",
        "ncs",
        "annI",
        "propL",
        "prevPropL",
        "annL",
        "imgL",
        "s_x",
    )
    def _log_any_change(self, change):
        logger.info(f"changed: {change.name}")

    # COMMANDS
    def set_image(self, im, ref=None, feats=None):
        logger.debug("set_image")
        self.image = im
        self.feats = im if feats is None else feats
        self.s_x, self.s_y, self.s_width, self.s_height = (
            0,
            0,
            im.shape[0],
            im.shape[1],
        )
        self.featsL = self.feats.copy()
        assert (
            self.feats.shape[0] == im.shape[0]
            and self.feats.shape[1] == im.shape[1]
        ), "feats should be HxWxF"
        self.ref = ref

        if ref is None:
            ref = np.zeros((im.shape[0], im.shape[1], 4))

        assert im.shape[:2] == ref.shape[:2], "Incompatible image and ref"
        image_metadata, image_buffer = binary_image(im)
        ref_metadata, ref_buffer = binary_image(ref)
        command = {
            "name": "image",
            "image": image_metadata,
            "ref": ref_metadata,
        }
        self.send(command, (image_buffer, ref_buffer))
        logger.debug("end set_image")

    def reset(self):
        logger.debug("reset")
        im, ref = self.image, self.ref
        if ref is None:
            ref = np.zeros((im.shape[0], im.shape[1], 4))
        self.propL = np.zeros_like(self.imgL).astype(np.uint8)

        assert im.shape[:2] == ref.shape[:2], "Incompatible image and ref"
        image_metadata, image_buffer = binary_image(im)
        ref_metadata, ref_buffer = binary_image(ref)
        command = {
            "name": "reset",
            "image": image_metadata,
            "ref": ref_metadata,
        }
        self.send(command, (image_buffer, ref_buffer))

    def useProposal(self):
        logger.debug("useProposal")
        self.prevPropL = np.zeros_like(self.prevPropL).astype(np.uint8)

        command = {
            "name": "useProposal",
        }
        self.send(command, None)
        self.run_tools(mode="prevPropL")

    def useReference(self):
        logger.debug("useReference")
        command = {
            "name": "useReference",
        }
        self.send(command, None)

    # Button functions
    def on_reset_button(self, b):
        logger.info("resetting...")
        self.reset()

    def on_useProposal_button(self, b):
        logger.info("setting proposal as annotation...")
        self.useProposal()

    def on_useReference_button(self, b):
        logger.info("setting reference as annotation...")
        try:
            self.useReference()
        except Exception as err:
            logger.exception("useReference failed")
            raise err

    @observe("s_x")
    def _layoutCoordsChanged(self, change):
        logger.debug("layoutCoordsChanged")
        logger.debug(self.s_x)
        logger.debug(self.s_y)
        logger.debug(self.s_height)
        logger.debug(self.s_width)

        try:
            xi, xf, yi, yf = (
                int(self.s_x),
                int(self.s_x + self.s_width),
                int(self.s_y),
                int(self.s_y + self.s_height),
            )
            pyi, pyf, pxi, pxf = (
                max(-yi, 0),
                max(0, yf - self.feats.shape[0]),
                max(-xi, 0),
                max(0, xf - self.feats.shape[1]),
            )
            self.featsL = np.pad(self.feats, ((pyi, pyf), (pxi, pxf), (0, 0)))[
                yi + pyi : yf + pyi, xi + pxi : xf + pxi
            ]
        except Exception as err:
            logger.debug(self.feats.shape)
            logger.debug(self.featsL.shape)
            logger.debug(f"{xi}, {xf}, {yi}, {yf}, {pyi}, {pyf}, {pxi}, {pxf}")
            logger.exception("layoutCoordsChanged failed")
            raise err

    # Event handlers
    @observe("tool")
    def _tool_changed(self, change):
        logger.info("changed tool")
        self.clear_canvases()
        self.run_tools(mode="prevPropL")

    @observe("size")
    def _size_changed(self, change):
        logger.info("size changed, calling _tool_changed")
        self.run_tools(mode="prevPropL")
        self.run_tools(mode="propL")

    @observe("annL")
    def _annL_changed(self, change):
        logger.info("annL changed, reshaping L if needed")
        self._update_layout(change)

    @observe("imgL")
    def _imgL_changed(self, change):
        logger.info("imgL changed, saving img")
        self._update_layout(change)

    def _update_layout(self, change):  # why doesn't this work?
        self.propL = (
            self.propL
            if self.propL.shape != (10, 10, 4)
            else np.zeros_like(self.annL).astype(np.uint8)
        )
        self.prevPropL = (
            self.prevPropL
            if self.prevPropL.shape != (10, 10, 4)
            else np.zeros_like(self.annL).astype(np.uint8)
        )

    @observe("annI")
    def _data_changed(self, change):
        # saturate non zero values
        new_data = np.repeat(
            (255 * (0 < change.new.sum(axis=2))).astype(np.uint8)[..., None],
            4,
            axis=2,
        )
        new_data[:, :, 1] = 0  # purplewash
        self.annI = new_data
        # automatically pushed to frontend

    @observe("pcs", "ncs")
    def _click_changed(self, change):
        logger.info("changed clicks")
        logger.debug(f"clicks: {self.pcs} {self.ncs}")
        try:
            self.run_tools(mode="propL")
        except Exception as err:
            logger.exception("error running tools")
            raise err

    # TOOLS (action is here)
    def clear_canvases(self):
        self.propL = np.zeros_like(self.imgL).astype(np.uint8)
        self.prevPropL = np.zeros_like(self.imgL).astype(np.uint8)

    def run_tools(self, mode):
        if self.tool == 0:
            pass  # lasso, handled by frontend
        elif self.tool == 1:
            pass  # brush, handled by frontend
        elif self.tool == 2:
            pass  # eraser, handled by frontend
        elif self.tool == 3:
            if mode == "prevPropL":
                self.iis_preview()  # (!) there are internal states
            elif mode == "propL":
                self.iis_propose()
        elif self.tool == 4:
            if mode == "prevPropL":
                self.superpix_preview()  # (!) there are internal states
            elif mode == "propL":
                self.superpix_propose()
        elif self.tool == 5:
            if mode == "prevPropL":
                self.cluster_preview()
            elif mode == "propL":
                self.cluster_propose()
        elif self.tool == 6:
            if mode == "prevPropL":
                self.cosine_preview()
            elif mode == "propL":
                self.cosine_propose()
        elif self.tool == 7:
            if mode == "prevPropL":
                self.gaussian_preview()
            elif mode == "propL":
                self.gaussian_propose()

    def cosine_preview(self):
        logger.info("cosine preview start")
        pass

    def cluster_preview(self):
        logger.info("cluster preview start")
        try:
            # clus = sklearn.mixture.GaussianMixture(
            #     n_components=self.size, max_iter=20
            # )
            clus = sklearn.cluster.KMeans(
                n_clusters=self.size, n_init=1, max_iter=20
            )

            segs = clus.fit_predict(
                self.featsL.reshape(-1, self.featsL.shape[-1])
            ).reshape(self.featsL.shape[:2])
            segs = scipy.signal.medfilt2d(
                segs, kernel_size=7
            )  # remove some noise
            segs = skimage.measure.label(segs+1, connectivity=1)
            segs = skimage.transform.resize(segs, self.imgL.shape[:2], order=0)
            logger.info("ended cluster fit")

            self.cluster_state["segs"] = segs
            logger.info(self.featsL.shape)
            logger.info(self.cluster_state["segs"].shape)
            logger.info(self.imgL.shape)
            seg_borders = skimage.segmentation.find_boundaries(segs)

            prev = np.zeros_like(self.imgL)
            prev[seg_borders] = np.array([0, 0, 255, 255])
            self.prevPropL = prev
            logger.info("cluster preview")
        except Exception as err:
            logger.exception("Failed running cluster")
            raise err

    # Superpix
    def superpix_preview(self):
        logger.info("superpix preview start")
        try:
            # self.segs = skimage.segmentation.slic(
            #     self.imgL,
            #     n_segments=self.size * 25,
            #     compactness=0.001,
            #     enforce_connectivity=1,
            # )  # internal variable used when clicking
            segs = skimage.segmentation.felzenszwalb(
                self.imgL, scale=self.size * 10, min_size=self.size * 10
            )  # internal variable used when clicking

            self.superpix_state["segs"] = segs

            seg_borders = skimage.segmentation.find_boundaries(segs)

            prev = np.zeros_like(self.imgL)
            prev[seg_borders] = np.array([0, 0, 255, 255])
            self.prevPropL = prev
            logger.info("superpix propose")
        except Exception as err:
            logger.error(self.imgL.shape)
            logger.exception("Failed running superpix")
            raise err

    def gaussian_preview(self):
        pass

    def gaussian_propose(self):
        logger.info("gaussian propose start")
        try:
            proposal = np.zeros_like(self.imgL, dtype=float)[
                ..., 0
            ]  # one channel only
            nfeatsL = skimage.transform.resize(
                self.featsL, self.imgL.shape[:2], order=0
            )
            nfeatsL = norm_fn(nfeatsL.astype(float))
            xs = nfeatsL[
                [pc[1] for pc in self.pcs], [pc[0] for pc in self.pcs]
            ].astype(
                float
            )  # N x C
            ys = nfeatsL[
                [nc[1] for nc in self.ncs], [nc[0] for nc in self.ncs]
            ].astype(
                float
            )  # M x C
            n, m = xs.shape[0], ys.shape[0]

            mu = ((xs.sum(0) - ys.sum(0)) / (n - m))[None]  # 1xC
            xsc, ysc = xs - mu, (
                ys - mu
            )  # / 100  # np.linalg.norm(ys - mu, axis=1)[:, None]
            sigma_inv = np.linalg.pinv((xsc.T @ xsc - ysc.T @ ysc) / (n - m))
            pixels = nfeatsL.reshape(-1, nfeatsL.shape[-1])
            distances = (
                (((pixels - mu) @ sigma_inv) * (pixels - mu))
                .sum(1)
                .reshape(nfeatsL.shape[:2])
            )
            distances = norm_fn(np.sqrt(distances - distances.min()))
            distances = skimage.exposure.equalize_hist(distances)
            proposal = (255 * (distances < (self.size / 100))).astype(
                np.uint8
            )  # thresholded, 1 channel
            proposal = np.repeat(proposal[..., None], 4, axis=2).astype(
                np.uint8
            )  # 4 channel
            self.propL = proposal
        except Exception as err:
            logger.exception("gaussian_propose failed")
            raise err

    def cosine_propose(self):
        logger.info("cosine propose start")
        try:
            proposal = np.zeros_like(self.imgL, dtype=float)[
                ..., 0
            ]  # one channel only
            self.featsL[self.featsL.sum(2) == 0] = np.eye(
                self.featsL.shape[-1]
            )[-1] * (self.featsL[self.featsL != 0].min())
            norm_factor = np.linalg.norm(self.featsL, axis=2)[..., None]
            norm_featsL = self.featsL.copy() / norm_factor
            norm_featsL = skimage.transform.resize(
                norm_featsL, self.imgL.shape[:2], order=0
            )
            for i, pc in enumerate(self.pcs):
                ref_feat = norm_featsL[pc[1], pc[0]][None, None]
                proposal = proposal + (ref_feat * norm_featsL).sum(
                    axis=2
                ) / len(self.pcs)
            proposal = norm_fn(skimage.exposure.equalize_hist(proposal))
            proposal = (255 * ((self.size / 100) < proposal)).astype(
                np.uint8
            )  # thresholded, 1 channel
            proposal = np.repeat(proposal[..., None], 4, axis=2).astype(
                np.uint8
            )  # 4 channel
            self.propL = proposal
        except Exception as err:
            logger.exception("cosine_propose failed")
            raise err

    def cluster_propose(self):
        logger.info("cluster propose start")
        try:
            segs = self.cluster_state["segs"]
            selected_labels = {}

            for pc in self.pcs:
                label = segs[pc[1], pc[0]]
                if label in selected_labels:
                    selected_labels[label] += 1
                else:
                    selected_labels[label] = 1
            for nc in self.ncs:
                label = segs[nc[1], nc[0]]
                if label in selected_labels:
                    selected_labels[label] -= 1
                else:
                    selected_labels[label] = -1

            prop = self.propL.copy()
            for label in selected_labels:
                if selected_labels[label] > 0:
                    prop[segs == label] = np.array(
                        [255, 255, 255, 255]
                    )  # green clicks white regions
                elif selected_labels[label] <= 0:
                    prop[segs == label] = np.array([0, 0, 0, 0])
            self.propL = prop
            logger.info("Created new proposal! (spix)")
            logger.info(f"shapes: {segs.shape} {self.propL.shape}")
        except Exception as err:
            logger.exception("Failed running cluster")
            raise err

    def superpix_propose(self):
        logger.info("annotating superpix...")
        try:
            segs = self.superpix_state["segs"]
            selected_labels = {}

            for pc in self.pcs:
                label = segs[pc[1], pc[0]]
                if label in selected_labels:
                    selected_labels[label] += 1
                else:
                    selected_labels[label] = 1
            for nc in self.ncs:
                label = segs[nc[1], nc[0]]
                if label in selected_labels:
                    selected_labels[label] -= 1
                else:
                    selected_labels[label] = -1

            prop = self.propL.copy()
            for label in selected_labels:
                if selected_labels[label] > 0:
                    prop[segs == label] = np.array(
                        [0, 255, 0, 255]
                    )  # green clicks green regions
                elif selected_labels[label] <= 0:
                    prop[segs == label] = np.array(
                        [0, 0, 0, 0]
                    )  # green clicks green regions
            self.propL = prop
            logger.info("Created new proposal! (spix)")
            logger.debug(f"shapes: {segs.shape} {self.propL.shape}")
        except Exception as err:
            logger.exception("Failed running superpix")
            raise err

    # IIS
    def iis_preview(self):
        pass

    def iis_propose(self):
        if len(self.pcs) + len(self.ncs) == 0:
            logger.info("no clicks yet...")
            return  # no clicks yet
        logger.info("running iis...")
        # get img and clicks in correct format
        img = self.imgL  # input image from current canvas
        assert (
            img.shape[2] == 4
        ), f"only rgba image suported but received {img.shape}"
        img = (
            torch.Tensor(norm_fn(rgba2rgb(img)) * 255)
            .permute(2, 0, 1)[None, ...]
            .float()
        )  # channel first
        H, W = img.shape[-2], img.shape[-1]
        target_H = math.ceil(H / 32) * 32
        target_W = math.ceil(W / 32) * 32
        padt, padb = (target_H - H) // 2, (target_H - H) // 2 + (
            target_H - H
        ) % 2
        padl, padr = (target_W - W) // 2, (target_W - W) // 2 + (
            target_W - W
        ) % 2
        img = torch.nn.functional.pad(img, (padl, padr, padt, padb))

        # assert (
        #     img.shape[-1] == img.shape[-2]
        # ), "image should be square for padding"
        # target_shape = math.ceil(img.shape[-1] / 32) * 32
        # padl = (target_shape - img.shape[-1]) // 2
        # padr = (target_shape - img.shape[-1]) // 2 + (
        #     target_shape - img.shape[-1]
        # ) % 2
        # img = torch.nn.functional.pad(img, (padl, padr, padl, padr))

        # fix clicks
        list_of_pcs, list_of_ncs = self.pcs, self.ncs
        pcs = [
            [[[padt + click[1], padl + click[0]]]] for click in list_of_pcs
        ] or [
            [[]]
        ]  # interaction, batch, click_number, (x,y)
        ncs = [
            [[[padt + click[1], padl + click[0]]]] for click in list_of_ncs
        ] or [[[]]]
        # pcs = [
        #     [[[padl + click[1], padl + click[0]]]] for click in list_of_pcs
        # ] or [
        #     [[]]
        # ]  # interaction, batch, click_number, (x,y)
        # ncs = [
        #     [[[padl + click[1], padl + click[0]]]] for click in list_of_ncs
        # ] or [[[]]]
        maxlen = max(len(pcs), len(ncs))
        pcs = pcs + [[[]]] * (maxlen - len(pcs))
        ncs = ncs + [[[]]] * (maxlen - len(ncs))

        # if first click, take as previous (in z) the current annotation annL
        if len(list_of_pcs) + len(list_of_ncs) == 1:  # first click
            use_annotation = True
            if use_annotation:
                logger.info("first click: use current mask as input")
                ref = self.annL
                assert (
                    ref.shape[2] == 4
                ), f"should be rgba image but is {ref.shape}"
                ref = norm_fn(ref.sum(axis=2)) * 255
                ref = torch.Tensor(ref)[None, None, ...].float()
                ref = torch.nn.functional.pad(ref, (padl, padr, padt, padb))
                # ref = torch.nn.functional.pad(ref, (padl, padr, padl, padr))
                self.iis_state["z"] = {
                    "prev": ref,
                    "prev_prediction": ref,
                    "prev_output": ref,
                }
            else:
                logger.info("first click: set prev to zero")
                ref = torch.zeros_like(img)[:, :1]
                self.iis_state["z"] = {
                    "prev": ref,
                    "prev_prediction": ref,
                    "prev_output": ref,
                }
        # if not first click, use last z
        else:
            logger.debug("using last z")

        try:
            if "prev_prediction" in self.iis_state["z"]:
                assert (
                    img.shape[-1]
                    == self.iis_state["z"]["prev_prediction"].shape[-1]
                    and img.shape[-2]
                    == self.iis_state["z"]["prev_prediction"].shape[-2]
                ), "image and prev should be the same size"
            elif "prev_output" in self.iis_state["z"]:
                assert (  # gto99
                    img.shape[-1]
                    == self.iis_state["z"]["prev_output"].shape[-1]
                    and img.shape[-2]
                    == self.iis_state["z"]["prev_output"].shape[-2]
                ), "image and prev should be square and same size"
            else:
                keys = self.iis_state["z"].keys()
                raise ValueError(
                    f"z is not what is expected, with keys {keys}"
                )
        except Exception as err:
            s1 = img.shape
            s2 = self.iis_state["z"]["prev_output"].shape
            s3 = self.iis_state["z"]["prev_prediction"].shape
            logger.error(f"failed with shapes {s1}, {s2}, {s3}")
            raise err
        try:
            y, z = self.iis_model(img, self.iis_state["z"], pcs, ncs)
            logger.debug(f"visualizing clicks: {pcs}, {ncs}")
            shapes = (
                to_np(img[0]).shape,
                to_np(self.iis_state["z"]["prev_prediction"][0])
                .squeeze()
                .shape,
                np.array(y[0][0].detach()).shape,
            )
            logger.debug(f"shapes: {shapes}")
            visualize_on_test(
                to_np(img[0]),
                to_np(
                    self.iis_state["z"]["prev_prediction"][0]
                ).squeeze(),  # channels last
                output=np.array(y[0][0].detach()),
                pcs=pcs,
                ncs=ncs,
                name="model_output",
                destdir=".",
            )
            pred = unpad(y[0][0], padl, padr, padt, padb).detach().numpy()
            # pred = unpad(y[0][0], padl, padr).detach().numpy()
            self.iis_state["z"] = z
        except Exception as err:
            logger.exception("IIS model failed :( ")
            logger.info(f"list of clicks {list_of_pcs}, {list_of_ncs}")
            raise err
        try:
            assert len(pred.shape) == 2, "one channel output"
        except AssertionError as err:
            raise err
        try:
            proposal = (255 * (0.5 < norm_fn(pred))).astype(
                np.uint8
            )  # thresholded, 1 channel
            proposal = np.repeat(proposal[..., None], 4, axis=2).astype(
                np.uint8
            )  # 4 channel
            self.propL = proposal
        except Exception as err:
            logger.exception("Could not generate proposal from output")
            raise err

        logger.info("Created new proposal! (iis)")


class FullSegmenter:
    def __init__(
        self, iis_model, curr_im, curr_ref=None, feats=None, layout_size=500
    ):
        w = segmenter(iis_model)
        w.layout.width = f"{layout_size}px"  # if too small, enlarge your image
        w.layout.height = f"{layout_size}px"
        w.set_image(curr_im, curr_ref, feats)

        alpha_slider = widgets.FloatSlider(
            value=0.3, min=0, max=1, step=0.05, description="Alpha mask"
        )
        widgets.jslink((alpha_slider, "value"), (w, "alpha"))

        size_slider = widgets.IntSlider(
            value=10, min=1, max=100, step=1, description="Tool slider"
        )
        widgets.jslink((size_slider, "value"), (w, "size"))

        tool_selector = widgets.RadioButtons(
            options=[
                "Lasso",
                "Brush",
                "Eraser",
                "IIS",
                "Superpixel",
                "Cluster",
                "Find similar - cosine",
                "Find similar - gaussian",
            ],
            description="Tool",
        )
        widgets.jslink((tool_selector, "index"), (w, "tool"))

        reset_button = widgets.Button(description="Reset", disabled=False)
        reset_button.on_click(w.on_reset_button)

        useProposal_button = widgets.Button(
            description="Use proposal", disabled=False
        )
        useProposal_button.on_click(w.on_useProposal_button)

        useReference_button = widgets.Button(
            description="Use reference", disabled=False
        )
        useReference_button.on_click(w.on_useReference_button)

        self.w = w
        self.widget = widgets.HBox(
            [
                w,
                widgets.VBox(
                    [
                        tool_selector,
                        alpha_slider,
                        size_slider,
                        reset_button,
                        useProposal_button,
                        useReference_button,
                    ]
                ),
            ]
        )

    def get_widget(self):
        return self.widget
