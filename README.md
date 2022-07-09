# The ultimate annotation tool in a Jupyter Notebook!

Follows:
- How to install
- How to use
- Examples

## Installation instructions
This project requires Python>=3.9, npm and nodejs. Their installation is on you.

If you want to create a fresh new virtual environment, open a terminal in this same place (ipyfan).
run (from inside ipyfan): `bash install.sh`. 

`install.sh` contains simple installation steps, thus using an already existing virtual environment is as simple as commenting the first few lines.

note: this works for bash. Yet, you can customize `install.sh` so it works with fish too. If it doesn't work, please open an issue.

## test with `example_single_image.ipynb`
```bash
source env_ipyfan/bin/activate 
cd example
jupyter notebook example_single_image.ipynb  
```

If everything goes well, you should be able to `Restart and run all` in the example notebook and start annotating the demo image.

## What is currently available?
### Tools
- image viewer with zoom and panning
- lasso 
- brush
- eraser
- superpixel
- iis = interactive image segmentation
### Controls
Everywhere:
- left click for everything

Only on simple tools:
- right/middle click to pan
- scroll to zoom

IIS:
- right click for "negative click" on IIS



## Buzzwords
This artificial intelligence (AI) assisted image annotation tool works locally and is awesome. It integrates great interactive image segmentation algorithms and a superpixel based segmentation. It is highly customizable by using Python (or Typescript). The objective is to quickly annotate segmentations or masks in images so to train deep learning computer vision models models.

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


