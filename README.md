# The ultimate image annotation tool in a Jupyter Notebook!

Follows:

- How to install
- How to use
- Examples

## Installation instructions

This project requires Python>=3.9, npm and nodejs (LTS, version 16). Their installation is on you.

If you want to create a fresh new virtual environment, open a terminal in this same place (ipyfan).
run (from inside ipyfan): `bash install.sh`.

`install.sh` contains simple installation steps, thus using an already existing virtual environment is as simple as commenting the first few lines.

note: this works for bash. Yet, you can customize `install.sh` so it works with fish too. If it doesn't work, please check that the appropiate versions of Python and Node are in use and open an issue.

## test with `example_single_image.ipynb`

```bash
source env_ipyfan/bin/activate  # activate the installation environment
cd example
jupyter notebook example_single_image.ipynb
```

If everything goes well, you should be able to `Restart and run all` in the example notebook and start annotating the demo images.

## What is currently available?


| Tool  | Left click (hold) | Right click (hold) | Wheel | Middle click (hold) | Tool slider | Description | 
| ----- | ----------------- | ------------------ | ----- | ------------------- | --- | --- |
| Lasso | (Draw contour)  | (Pan) | Zoom in/out | (Pan) | - | - |
| Brush | Paint around cursor | (Pan) | Zoom in/out | (Pan) | Change brush size | - |
| Eraser | Erase around cursor | (Pan) | Zoom in/out | (Pan) | Change eraser size | The only way to reduce violet mask |
| IIS | Positive click  | Negative click | - | - | - | - |
| Superpixels | Add positive click/(scribble)  | Add negative click/(scribble) | - | - | Change superpixel scale | A segment is proposed if #pos > #neg |
| Clustering | Add positive click/(scribble)  | Add negative click/(scribble) | - | - | Change number of clusters | A segment is proposed if #pos > #neg |
| Cosine | Add positive click/(scribble)  | - | - | - | Change mean cosine similarity threshold | - |
| Gaussian | Add positive click/(scribble)  | Add negative click/(scribble) if #neg < #pos-1 | - | - | Change threshold on Mahalanobis distance | Using large #neg causes inestability |



#### Remarks
- Final masks are violet.
- Masks that are not violet are proposals (or reference).
- Proposals (reference) are made violet via the "Use proposal" ("Use reference") button.
- All modifications (except eraser) are only additive.
- Methods are run after a tool slider or selected tool change.
- For more detail, read the `user_guide.md`


## Buzzwords

This artificial intelligence (AI) assisted image annotation tool works locally and is awesome. It integrates great interactive image segmentation algorithms and a superpixel based segmentation. It is highly customizable by using Python (or Typescript). The objective is to quickly annotate segmentations or masks in images so to train deep learning computer vision models. We use Pytorch, Scikit-learn, scikit image, timm, ipywidgets and traitlets.

# Acknowledgment

Inspired by [Ian Hunt-Isaak's annotation tool](https://github.com/ianhi/ipysegment)

# Dev

## How to see your Typescript change:

To continuously monitor the project for changes and automatically:

```bash
npm run watch
```

After a change wait for the build to finish and then refresh your browser and the changes should take effect.

## How to see your Python change:

Restart the kernel of the notebook.

## To-do / feature requests

- display image filename
- load previously annotated mask as reference
- increase display size
- create universal `read_image` function
- allow pan and zoom while using IIS (and more generally, advanced tools)
- avoid crashes if neg pos clicking the same place
- make gaussian estimation more robust
