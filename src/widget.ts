// Copyright (c) Ian Hunt-Isaak
// Distributed under the terms of the Modified BSD License.

import { DOMWidgetModel, DOMWidgetView, ISerializers, Dict } from '@jupyter-widgets/base';

import { MODULE_NAME, MODULE_VERSION } from './version';

import * as pako from 'pako';
import { data_union_serialization, getArray, listenToUnion } from 'jupyter-dataserializers';


// Import the CSS
import '../css/widget.css';
import ndarray from 'ndarray';
import { LassoTool } from './tools/lasso';
import { BrushTool } from './tools/brush';
import { EraserTool } from './tools/eraser';
import { IISTool } from './tools/iis';
import { SuperpixTool } from './tools/superpix';

function serializeImageData(image: ImageData) {
  console.log('serializing');
  const data = pako.deflate(new Uint8Array(image.data.buffer));
  return { width: image.width, height: image.height, data: data };
}

function deserializeImageData(dataview: DataView | null) {
  return null;
}

export class segmentModel extends DOMWidgetModel {

  defaults() {
    return {
      ...super.defaults(),
      _model_name: segmentModel.model_name,
      _model_module: segmentModel.model_module,
      _model_module_version: segmentModel.model_module_version,
      _view_name: segmentModel.view_name,
      _view_module: segmentModel.view_module,
      _view_module_version: segmentModel.view_module_version,
    };
  }

  static serializers: ISerializers = {
    ...DOMWidgetModel.serializers,
    _labels: {
      serialize: serializeImageData,
      deserialize: deserializeImageData,
    },
    annI: data_union_serialization,
    annL: data_union_serialization,
    imgL: data_union_serialization,
    propL: data_union_serialization,
    prevPropL: data_union_serialization,
  };

  initialize(attributes: any, options: any) {
    super.initialize(attributes, options);

    this.imgICanvas = document.createElement('canvas');
    this.refICanvas = document.createElement('canvas');
    this.annICanvas = document.createElement('canvas');
    this.propICanvas = document.createElement('canvas');
    this.propLCanvas = document.createElement('canvas');

    this.imgIContext = getContext(this.imgICanvas);  // image
    this.refIContext = getContext(this.refICanvas);  // reference segmentation
    this.annIContext = getContext(this.annICanvas);  // user segmentation
    this.propIContext = getContext(this.propICanvas);  // proposed segmentation
    this.propLContext = getContext(this.propLCanvas);  // proposed segmentation on layout

    for (const ctx of [this.imgIContext, this.refIContext, this.annIContext, this.refIContext, this.propIContext, this.propLContext,]) {
      ctx.imageSmoothingEnabled = false;
      ctx.lineWidth = 2;
    }

    listenToUnion(this, 'annI', this._updateAnnIData.bind(this), true);  // orig
    listenToUnion(this, 'propL', this._updatePropLData.bind(this), true);
    listenToUnion(this, 'prevPropL', this._updatePrevPropLData.bind(this), true);
    // don't listen to: imgI, refI, propI

    // listen to commands from backend
    this.on('msg:custom', this.onCommand.bind(this));

    // define default values
    this.alpha = .3;
    this.tool = 0;
    this.toolSize = 10;

    this.listenTo(this, 'change:alpha', this._alpha_changed);
    this.listenTo(this, 'change:tool', this._tool_changed);
    this.listenTo(this, 'change:size', this._size_changed);

    this._clear()
  }

  //////////////////////////////////////////////////////////////////////////////
  // tool changing functions
  private _alpha_changed() {
    this.alpha = this.get('alpha');
    this._forEachView((view) => {
      view.redraw();
    });
    console.log('alpha')
  }

  private _tool_changed() {
    this._clear()  // clears everything but img, ref, ann
    this.tool = this.get('tool');
    this.set('tool', this.tool)

    this._forEachView((view) => {
      view.tools[this.tool].onStart();
      view.redraw();
      view.pushAnnToBackend();
      view.pushImgLToBackend();  // sync what we are seeing (remains fixed until tool change)
    });

    this.save_changes()
    console.log('tool')
  }

  private _size_changed() {
    this.toolSize = this.get('size');
    this._forEachView((view) => {
      view.debouncedPushToBackend();
      view.redraw();
    });
    this.set('size', this.toolSize)
    this.save_changes()
    console.log('size')
  }

  //////////////////////////////////////////////////////////////////////////////
  ///////////////  COMMANDS  ///////////////////////////////////////////////////

  private onCommand(command: any, buffers: any) {
    // Change the main image or the reference but keep the current annotation
    if (command.name === 'image') {
      this.putImageData(command.image, command.ref, buffers);
      this._updateAnnIData();
      this._forEachView((view) => {
        view.resize();
        view.redraw();
        view.pushAnnToBackend();
        view.pushImgLToBackend();  // sync what we are seeing (remains fixed until tool change)
      });
    }
    // Reset the current annotation and change the different canvas
    else if (command.name === 'reset') {
      this._clear()
      this.annIContext.clearRect(0, 0, this.annICanvas.width, this.annICanvas.height);
      this.putImageData(command.image, command.ref, buffers);  // update img and ref
      this._forEachView((view) => {
        view.pushAnnToBackend()
        view.pushImgLToBackend();  // sync what we are seeing (remains fixed until tool change)
      });
      console.log('end reset')
    }
    else if (command.name === 'useProposal') {
      // put propIContext into annIContext
      this.annIContext.drawImage(this.propICanvas, 0, 0);
      this._clear()
      this._forEachView((view) => {
        view.pushAnnToBackend();
        view.pushImgLToBackend();  // sync what we are seeing (remains fixed until tool change)
      });
      console.log('useProposal')
    }
    else if (command.name === 'useReference') {
      this.annIContext.drawImage(this.refICanvas, 0, 0);
      this._forEachView((view) => {
        view.resize();
        view.redraw();
        view.pushAnnToBackend();
      });
      console.log('useReference')
    }

  }

  //////////////////////////////////////////////////////////////////////////////
  private _clear(): void {  // clears proposals and previews
    this.pcs = [];  // always clear list of clicks
    this.ncs = [];
    // clear all canvases (except img,ref,ann)
    this.propIContext.clearRect(0, 0, this.propICanvas.width, this.propICanvas.height);
    this.propLContext.clearRect(0, 0, this.propLCanvas.width, this.propLCanvas.height);

    this._forEachView((view) => {
      view.resetView();
      view.pushImgLToBackend();  // sync what we are seeing (first time)
      view.pushAnnToBackend();  // sync what we are seeing (first time)
    })
    this.set('pcs', this.pcs)
    this.set('ncs', this.ncs)
    this.save_changes()
    console.log('clear')
  }

  private _updateData(fieldName: string, destContext: CanvasRenderingContext2D): void {
    const img = getArray(this.get(fieldName));
    if (img) {
      const imageData = new ImageData(
        new Uint8ClampedArray(img.data),
        img.shape[1],
        img.shape[0]
      );
      destContext.putImageData(imageData, 0, 0);
      console.log(destContext)
    }
  }

  private _updateAnnIData() {
    this._updateData('annI', this.annIContext);
    this._forEachView((view) => {
      view.redraw();
    });
    console.log('_updateAnnIData');
  }

  private _updatePropLData() {
    this._updateData('propL', this.propLContext);
    this.propIContext.clearRect(0, 0, this.propICanvas.width, this.propICanvas.height);
    this._forEachView((view) => {
      this.propIContext.drawImage(  // put prop on bigger prop
        this.propLCanvas,
        0,
        0,
        this.propLCanvas.width,
        this.propLCanvas.height,
        view._Sx,
        view._Sy,
        view._sWidth,
        view._sHeight
      );
      view.redraw();
    });
    console.log('_updatePropLData');
  }

  private _updatePrevPropLData() {
    this._forEachView((view) => {
      this._updateData('prevPropL', view.previewLContext);
    })
    console.log('_updatePrevPropLData');
  }


  pushImageToBackend(canvas: HTMLCanvasElement, context: CanvasRenderingContext2D, name: string) {
    // Update data
    const h = canvas.height;
    const w = canvas.width;
    const imgData = context.getImageData(0, 0, w, h);
    const nd = ndarray(imgData.data, [h, w, 4]);
    this.set(name, nd);
    this.save_changes();
  }

  putImageData(bufferMetadataIm: any, bufferMetadataRef: any, buffers: any) {
    console.log('putImageData start');
    this.imgWidth = bufferMetadataIm.shape[1];
    this.imgHeight = bufferMetadataIm.shape[0];

    const dataIm = new Uint8ClampedArray(buffers[0].buffer);
    const dataRef = new Uint8ClampedArray(buffers[1].buffer);
    const imageData = new ImageData(dataIm, this.imgWidth, this.imgHeight);
    const refData = new ImageData(dataRef, this.imgWidth, this.imgHeight);
    this.resizeDataCanvas(`${this.imgWidth}px`, `${this.imgHeight}px`);
    this.imgIContext.putImageData(imageData, 0, 0);
    this.refIContext.putImageData(refData, 0, 0);
    this._forEachView((view) => {
      view.resize();
      view.redraw();
    });
  }

  private resizeDataCanvas(width: string, height: string) {
    this.imgICanvas.setAttribute('width', width);
    this.imgICanvas.setAttribute('height', height);
    this.annICanvas.setAttribute('width', width);
    this.annICanvas.setAttribute('height', height);
    this.refICanvas.setAttribute('width', width);
    this.refICanvas.setAttribute('height', height);
    this.propICanvas.setAttribute('width', width);
    this.propICanvas.setAttribute('height', height);
  }

  //again from ipycanvas
  private _forEachView(callback: (view: segmentView) => void) {
    for (const view_id in this.views) {
      this.views[view_id].then((view: segmentView) => {
        callback(view);
      });
    }
  }
  static model_name = 'segmentModel';
  static model_module = MODULE_NAME;
  static model_module_version = MODULE_VERSION;
  static view_name = 'segmentView'; // Set to null if no view
  static view_module = MODULE_NAME; // Set to null if no view
  static view_module_version = MODULE_VERSION;

  imgICanvas: HTMLCanvasElement;
  annICanvas: HTMLCanvasElement;
  refICanvas: HTMLCanvasElement;
  propICanvas: HTMLCanvasElement;
  propLCanvas: HTMLCanvasElement;

  imgIContext: CanvasRenderingContext2D;
  annIContext: CanvasRenderingContext2D;
  refIContext: CanvasRenderingContext2D;
  propIContext: CanvasRenderingContext2D;
  propLContext: CanvasRenderingContext2D;  // one of these is useless

  imgWidth: number;
  imgHeight: number;

  alpha: number;
  tool: number;
  toolSize: number;
  pcs: number[][];
  ncs: number[][];

  views: Dict<Promise<segmentView>>;
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

export class segmentView extends DOMWidgetView {
  render(): void {
    // crates container that overlays display and preview Canvases
    const container = document.createElement('div');
    this.displayLCanvas = document.createElement('canvas');
    this.previewLCanvas = document.createElement('canvas');
    this.previewLCanvas.classList.add('preview');
    container.setAttribute('position', 'relative');
    this.el.appendChild(container);
    this.el.classList.add('segment-container');
    container.appendChild(this.displayLCanvas);
    container.appendChild(this.previewLCanvas);
    this.previewLContext = getContext(this.previewLCanvas);
    this.displayLContext = getContext(this.displayLCanvas);

    // act on user input from the preview Canvas
    this.previewLCanvas.addEventListener('mouseup', this._mouseUp);
    this.previewLCanvas.addEventListener('mousedown', this._mouseDown);
    this.previewLCanvas.addEventListener('mousemove', this._mouseMove);
    this.previewLCanvas.addEventListener('wheel', this._wheel);
    this.previewLCanvas.addEventListener('contextmenu', (e) => {
      e.preventDefault();
      e.stopPropagation();
    });

    // draw 
    this._sHeight = this.model.annICanvas.height;
    this._sWidth = this.model.annICanvas.width;
    this.resize();
    this.redraw();
    // sync
    this.pushAnnToBackend()
    this.pushImgLToBackend();  // sync what we are seeing (remains fixed until tool change)
  }

  //////////////////////////////////////////////////////////////////////////////
  ////////////////////  VISUALIZATIONS  ////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////

  resize() {
    const aspectRatio = this.model.imgWidth / this.model.imgHeight;
    if (aspectRatio > 1) {
      this.displayWidth = parseFloat(this.model.get('layout').get('width'));
      this.displayHeight = this.displayWidth / aspectRatio;
      this.intrinsicZoom = this.model.imgWidth / this.displayWidth;
    } else {
      this.displayHeight = parseFloat(this.model.get('layout').get('height'));
      this.displayWidth = this.displayHeight * aspectRatio;
      this.intrinsicZoom = this.model.imgHeight / this.displayHeight;
    }
    this.displayLCanvas.setAttribute('width', `${this.displayWidth}px`);
    this.displayLCanvas.setAttribute('height', `${this.displayHeight}px`);
    this.previewLCanvas.setAttribute('width', `${this.displayWidth}px`);  // this clears the canvas!
    this.previewLCanvas.setAttribute('height', `${this.displayHeight}px`);
    this.model.propLCanvas.setAttribute('width', `${this.displayWidth}px`);
    this.model.propLCanvas.setAttribute('height', `${this.displayHeight}px`);
    this.displayLContext.imageSmoothingEnabled = false;
    this.previewLContext.imageSmoothingEnabled = false;
    this.model.propLContext.imageSmoothingEnabled = false;
  }

  redraw(): void {
    // at some point should consider: https://developer.mozilla.org/en-US/docs/Web/API/Canvas_API/Tutorial/Optimizing_canvas
    // to not always scale the image. But maybe not necessary, I still seem to get 60 fps
    // according to the firefox devtools performance thing
    this.displayLContext.clearRect(0, 0, this.displayLCanvas.width, this.displayLCanvas.height);
    this._drawInDisplayAlias(this.model.imgICanvas);
    this.displayLContext.globalAlpha = this.model.alpha;
    this._drawInDisplayAlias(this.model.annICanvas);
    this._drawInDisplayAlias(this.model.refICanvas);
    this._drawInDisplayAlias(this.model.propICanvas);
    this.displayLContext.globalAlpha = 1;
  }





  _drawClicks(): void {
    // draws the clicks on the preview canvas
    // for each click, draw a circle on the class canvas
    var clickColor = 'rgb(0, 255, 0)';
    for (let i = 0; i < this.model.pcs.length; i++) {
      var click = this.model.pcs[i];
      // little square
      this.previewLContext.fillStyle = clickColor;
      this.previewLContext.fillRect(click[0] - 5, click[1] - 5, 10, 10);
    }
    if (this.model.tool != 4) {
      var clickColor = 'rgb(255, 0, 0)';
      for (let i = 0; i < this.model.ncs.length; i++) {
        var click = this.model.ncs[i];
        this.previewLContext.fillStyle = clickColor;
        this.previewLContext.fillRect(click[0] - 5, click[1] - 5, 10, 10);
      }
    }
  }


  //////////////////////////////////////////////////////////////////////////////
  ////////////////////  INTERACTIONS   /////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////


  canvasCoords(e: MouseEvent): [number, number] {
    let mouseX = e.offsetX;
    let mouseY = e.offsetY;
    mouseX -= this.previewLCanvas.offsetLeft;
    mouseY -= this.previewLCanvas.offsetTop;
    return [mouseX, mouseY];
  }

  //////////////////////////////////////////////////////////////////////////////
  ////////////////////  INTERACTIONS: wheel  ///////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////

  // gotta be an arrow function so we can keep on
  // using this to refer to the Drawing rather than
  // the clicked element
  private _wheel = (e: WheelEvent): void => {
    if (this.model.tool !== 3 && this.model.tool !== 4) {
      let scale = 1;
      if (e.deltaY < 0) {
        scale = 1.1;
      } else if (e.deltaY > 0) {
        scale = 1 / 1.1;
      }
      const [x, y] = this.canvasCoords(e);
      const left = (x * this.intrinsicZoom) / this.userZoom;  // position of mouse in the canvas measured in pixels of the image
      const down = (y * this.intrinsicZoom) / this.userZoom;
      const X = this._Sx + left;  // position of mouse in the image measured in pixels of the image 
      const Y = this._Sy + down;
      this.userZoom *= scale;
      this._sWidth /= scale;
      this._sHeight /= scale;
      const newLeft = left / scale;
      const newDown = down / scale;
      this._Sx = X - newLeft;
      this._Sy = Y - newDown;

      this.resize()
      this.redraw()
      e.preventDefault();
      this.debouncedPushToBackend()

    }
  };



  //////////////////////////////////////////////////////////////////////////////
  ////////////////////  INTERACTIONS: mouse Down  //////////////////////////////
  //////////////////////////////////////////////////////////////////////////////


  private mouseDownPan(e: MouseEvent): void {
    const [mouseX, mouseY] = this.canvasCoords(e);
    this._panStartX = this._Sx + (mouseX * this.intrinsicZoom) / this.userZoom;
    this._panStartY = this._Sy + (mouseY * this.intrinsicZoom) / this.userZoom;
    this.panning = true;
    e.preventDefault();
  }

  private mouseMovePan(e: MouseEvent): void {
    const [mouseX, mouseY] = this.canvasCoords(e);
    this._Sx = this._panStartX - (mouseX * this.intrinsicZoom) / this.userZoom;
    this._Sy = this._panStartY - (mouseY * this.intrinsicZoom) / this.userZoom;
    this.redraw();

  }

  private mouseUpPan(e: MouseEvent): void {
    this.panning = false;
    this.pushAnnToBackend();
    this.pushImgLToBackend();
  }


  private _mouseDown = (e: MouseEvent): void => {
    if (e.button === 1 || (e.button === 2 && this.model.tool !== 3 && this.model.tool !== 4)) {  // middle click = pan
      this.mouseDownPan(e);  // pan with middle button or, if outside iis/spix mode, with right button too 
    }
    else if ((e.button === 0) || (e.button === 2 && this.model.tool === 3)) {  // left click
      this.tools[this.model.tool].mouseDown(e, this);
    }
  }


  private _mouseMove = (e: MouseEvent): void => {
    if (this.panning) {
      this.mouseMovePan(e);
    }
    else {  // lasso implementation
      this.tools[this.model.tool].mouseMove(e, this);
    }
  };

  private _mouseUp = (e: MouseEvent): void => {
    if (this.panning) {
      this.mouseUpPan(e);
    }
    else {
      this.tools[this.model.tool].mouseUp(e, this)
    }
  };


  resetView(): void {
    // clear all canvases
    console.log('resetView...')
    this.previewLContext.clearRect(0, 0, this.previewLCanvas.width, this.previewLCanvas.height);
    this.displayLContext.clearRect(0, 0, this.displayLCanvas.width, this.displayLCanvas.height);
    this.resize();
    this.redraw();
  }

  pushAnnToBackend() {  // both I and L, resets displayLContext
    this.displayLContext.clearRect(0, 0, this.displayLCanvas.width, this.displayLCanvas.height);
    this._drawInDisplayAlias(this.model.annICanvas)
    this.model.pushImageToBackend(this.displayLCanvas, this.displayLContext, 'annL');
    this.model.pushImageToBackend(this.model.annICanvas, this.model.annIContext, 'annI');
    this.redraw();
  }


  pushImgLToBackend() {  // resets displayLContext
    this.displayLContext.clearRect(0, 0, this.displayLCanvas.width, this.displayLCanvas.height);
    this._drawInDisplayAlias(this.model.imgICanvas)
    this.model.pushImageToBackend(this.displayLCanvas, this.displayLContext, 'imgL');
    this.redraw()
  }

  debouncedPushToBackend = debounce(() => {
    this.pushAnnToBackend();
    this.pushImgLToBackend();
  }, 300)



  private _drawInDisplayAlias(sourceCanvas: HTMLCanvasElement): void {
    this.displayLContext.drawImage(
      sourceCanvas,
      this._Sx, // sx
      this._Sy, // sy
      this._sWidth,
      this._sHeight,
      0, // dx
      0, // dy
      this.displayLCanvas.width,
      this.displayLCanvas.height
    );
  }

  _drawInAnnAlias(sourceCanvas: HTMLCanvasElement): void {
    this.model.annIContext.drawImage(  // put preview on class canvas
      sourceCanvas,
      0,
      0,
      sourceCanvas.width,
      sourceCanvas.height,
      this._Sx,
      this._Sy,
      this._sWidth,
      this._sHeight
    );
  }


  model: segmentModel;
  displayLCanvas: HTMLCanvasElement;
  previewLCanvas: HTMLCanvasElement;
  displayLContext: CanvasRenderingContext2D;
  previewLContext: CanvasRenderingContext2D;
  private panning: boolean;
  private userZoom = 1;
  private intrinsicZoom = 1;
  _Sx = 0;
  _Sy = 0;
  _sWidth = 0;
  _sHeight = 0;
  private _panStartX = 0;
  private _panStartY = 0;
  private displayWidth = 1000;
  private displayHeight = 1000;

  // private toolIsActive: boolean;
  tools = [new LassoTool(), new BrushTool(), new EraserTool(), new IISTool(), new SuperpixTool];
}

// taken from ipycanvas
function getContext(canvas: HTMLCanvasElement) {
  const context = canvas.getContext('2d');
  if (context === null) {
    throw 'Could not create 2d context.';
  }
  return context;
}

const debounce = (fn: Function, ms = 300) => {
  let timeoutId: ReturnType<typeof setTimeout>;
  return function (this: any, ...args: any[]) {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => fn.apply(this, args), ms);
  };
};
