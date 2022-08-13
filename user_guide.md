
# User guide

## Tips
- Add with care so you don't have to erase.
- Experiment which tool works best for your problem. From most to least sophisticated: IIS, gaussian, cosine, superpixel, clustering, lasso, brush. 
- Modify the data management to your liking
- Change the alpha and use the tool slider (not for Lasso/IIS)!
- Provide feedback by sending an email to marchesoniacland@gmail.com with "[tool feedback] (...)" as subject



## Concepts
- **Violet mask**: the final mask to be saved, i.e. the mask you are working on

## Summary

### Tools
- image viewer with zoom and panning
- Lasso 
- Brush
- Eraser
- Superpixels
- IIS = interactive image segmentation
- Unsupervised segmentation (clustering)
- Threshold on mean cosine similarity
- Gaussian fit


### Controls
Everywhere:
- left click for everything

Only on simple tools:
- right/middle click to pan
- scroll to zoom

Other tools:
- right click for "negative click"

All tools except Lasso and IIS:
- tool slider

Final masks are violet.
Masks that are not violet are proposals.
Proposals are made violet via the "Use proposal" option.
All modifications (except eraser) are only additive.


## Detailed

### Data
Data loading and data management is responsability of the user through the notebook.
The `FullSegmenter` contains the `segmenter` and some buttons. When calling `segmenter.set_image(im, ref, feats)` you are expected to give an RGB image that will be internally converted to RGBA.
Both `ref` and `feats` can be `None`. The reference `ref` is supposed to be the existing annotation (a binary mask), which you can make violet when clicking `Use reference`, and then edit it. The `feats` is expected to be the same size as the image, but might be multichannel. Here you should load your full hyperspectral image if you have one or an feature map with one feature vector per pixel. If you don't provide `feats`, they will simply be the RGB vector.

### Basic tools

#### Zooming and panning
You can use the wheel for zooming in/out if you have a mouse (or equivalent). For panning, dragging the right click or the middle click should work. This is valid for all the basic tools in this section.

In the current version we can not use panning and zooming on the advanced features because it is hard to maintain everything aligned and debounce the backend model running. In the meantime, change to a basic tool, move/zoom and come back to your favourite tool.

#### Lasso
Hold down the left click and create a countour. The last point will be joined with the first one with a straight line, you should be able to visualize it and get a better intuition by using it. 
Useful for great borders when zooming in or for big areas. Tool slider does nothing.

#### Brush
You know this one from MS Paint. It is square because the Canvas API is limited. You can change the size with the slider. Activate

#### Eraser
Just what you expect. It is the opposite of the brush. It is invisible, sorry. This is the most important tool because is the only you can use to correct mistakes on the violet mask.

### Advanced tools
- These run on the "backend".
- All of them, except IIS, accept scribbles (holding down the mouse).
- They are activated whenever the tool slider or the selected tool change. Then, if you clicked "Reset", you won't see much.
- Every code change is reflected only after kernel restart.

#### IIS
IIS stands for "interactive image segmentation". This is a neural network that takes as input the RGB image, the positive / negative clicks, and the previous mask. It was trained to perform well on COCO dataset, which comprises natural images (images of dogs, buildings, etc.). 

Some considerations:
- The violet mask is the initial mask, you will need some clicks to preserve it.
- The tool works best when clicking on the middle of the largest misclassified region.
- IIS will perform better on natural images than on other domains.
- Extra image features are not used.
- Positive click = left click, negative click = right click. Proposal should be accepted with `Use proposal` button. Tool slider is useless.

We use the implementation from [ritm](https://github.com/saic-vul/ritm_interactive_segmentation).

#### Superpixel
This tool groups pixels that are close together in color and close together spatially. Regions with more positive than negative clicks will be proposed. Tool slider changes the density or number of the possible regions.

We use `skimage` Felzenszwalb's algorithm, but you can use SLIC if you want: go to `ipyfan/segment.py` and comment/uncomment the first few `superpix_preview` lines.

#### Clustering
This is clustering over the feature space, with almost no notion of spatial relations (except some median filtering). It is done with KMeans because it is faster, but you can go to `ipyfan/segment.py` and change the first few lines in `cluster_preview` if you want to use a GMM instead. The same criteria than in Superpixel is used for proposal generation. The number of clusters is changed with the tool slider.

In the current implementation all connected components are different segments. If you want to "find similar", i.e. click on some part of the image and discover those similar parts in the rest of the image, you should comment out the line `segs = skimage.measure.label(segs+1, connectivity=1)` in `ipyfan/segment.py`.

#### Cosine
This tool uses the cosine similarity between features, i.e. the dot product between the normalized feature vectors. It does not accept negative clicks. If you do many clicks, the mean cosine distance will be used. The tool slider controls the threshold used.

#### Gaussian
This tool finds a gaussian in the feature space that best fits the positive points. We developed some equations that allow negative points to also be considered, but they introduce some mathematical stability issues. Then:
- Use as many positive points as you want and use the tool slider to find a good threshold.
- You can correct / modify the estimation by placing a few negative points. 
- Clicking near the positive mean (in the feature space) and having low number of negative points favors stability.
- If estimation diverges, place more positive points or just reset.
- Number of negative points is forced to be smaller than the number of positive points.


### Internal variables
You can access variables from the `segmenter`. For instance, from most to least useful:
- `segmenter.annI` is the annotation you are creating
- `segmenter.annL` is the annotation visible in the current window
- `segmenter.propL` is the proposal at he current window
- `segmenter.prevPropL` is the extra stuff such as segment borders at the current window

## Pro tips
- If you have hyperspectral data or any other per pixel features, add them and use gaussian, cosine or clustering tools. E.g. experiment with `skimage.feature.multiscale_basic_features`.

