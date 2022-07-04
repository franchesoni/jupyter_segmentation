# The ultimate annotation tool in a Jupyter Notebook!

See the example notebook provided in `example/`.

## Installation instructions
This project requires npm and nodejs. Install them if you don't already have them.

Open a terminal in this same place (ipyfan).
run (from inside ipyfan): `bash installation.sh`

This should be it, but if you don't want the interactive image segmentation (IIS) tool, you should comment the corresponding line in the installation script.

## test with `example.ipynb`
```bash
source env_ipyfan/bin/activate 
jupyter notebook example/example.ipynb  
```

If everything goes well, you should be able to `Run all` in the `example.ipynb` and start annotating the demo image.

## What is currently available?
### Tools
- image viewer with zoom and panning
- lasso 
- brush
- eraser
- superpixel
- iis = interactive image segmentation
### Controls
- *right/middle click to pan
- *scroll to zoom
- left click for everything
- right click for "negative click" on IIS

*when outside superpixel or iis

## Dev
### How to see your Typescript change:
To continuously monitor the project for changes and automatically:
```bash
npm run watch
```
After a change wait for the build to finish and then refresh your browser and the changes should take effect.

### How to see your Python change:
Restart the kernel of the notebook.


# Acknowledgment
Inspired by [Ian Hunt-Isaak's annotation tool](https://github.com/ianhi/ipysegment)