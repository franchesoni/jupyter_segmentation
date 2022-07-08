#!/usr/bin/env python
# coding: utf-8

# Copyright (c) Ian Hunt-Isaak.
# Distributed under the terms of the Modified BSD License.

"""
TODO: Add module docstring
"""

from ipywidgets import DOMWidget, Output

from ._frontend import module_name, module_version
from .utils import binary_image, norm_fn, unpad, rgba2rgb, to_np
from numpy import frombuffer, uint8, copy
import zlib
from ipydatawidgets import (
    shape_constraints,
    DataUnion,
    data_union_serialization,
)
import numpy as np

import skimage.segmentation
import skimage.transform
import skimage.filters
import skimage.morphology

import math
import torch

# iislib_path = Path(__file__).parent.parent.parent / "iislib/iislib"
# sys.path.append(str(tests_path))
# tests_path = Path(__file__).parent.parent.parent / "iislib/tests"
# sys.path.append(str(tests_path))


# sys.path.append("/home/franchesoni/mine/creations/phd/projects/stego/STEGO/src/")
# from stego_apply import stego_apply
# import torchvision.transforms as T
# from PIL import Image

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
    )
    def _log_any_change(self, change):
        logger.info(f"changed: {change.name}")

    # COMMANDS
    def set_image(self, im, ref=None):
        logger.debug("set_image")
        self.image = im
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

    # Event handlers
    @observe("tool")
    def _tool_changed(self, change):
        logger.info("changed tool")
        self.clear_canvases()
        self.run_tools(mode="prevPropL")
        # self.run_tools(mode='propL')

    @observe("size")
    def _size_changed(self, change):
        logger.info("size changed, calling _tool_changed")
        self.run_tools(mode="prevPropL")

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
                self.imgL, scale=self.size * 20, min_size=self.size * 20
            )  # internal variable used when clicking
            # segs = self.stego_apply_casted(self.imgL)

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

    def superpix_propose(self):
        logger.info("annotating superpix...")
        try:
            segs = self.superpix_state["segs"]
            selected_labels = []
            for pc in self.pcs:
                selected_labels.append(segs[pc[1], pc[0]])
            prop = self.propL.copy()
            for label in selected_labels:
                prop[segs == label] = np.array(
                    [0, 255, 0, 255]
                )  # green clicks green regions
            self.propL = prop
            logger.info("Created new proposal! (spix)")
            logger.info(f"shapes: {segs.shape} {self.propL.shape}")
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
                keys = self.iis_state['z'].keys()
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
    def __init__(self, iis_model, curr_im, curr_ref, layout_size=500):
        w = segmenter(iis_model)
        w.layout.width = f"{layout_size}px"  # if too small, enlarge your image
        w.layout.height = f"{layout_size}px"
        w.set_image(curr_im, curr_ref)

        alpha_slider = widgets.FloatSlider(
            value=0.3, min=0, max=1, step=0.05, description="alpha mask"
        )
        widgets.jslink((alpha_slider, "value"), (w, "alpha"))

        size_slider = widgets.IntSlider(
            value=10, min=1, max=100, step=1, description="tool size"
        )
        widgets.jslink((size_slider, "value"), (w, "size"))

        tool_selector = widgets.RadioButtons(
            options=["lasso", "brush", "eraser", "iis", "superpixel"],
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
