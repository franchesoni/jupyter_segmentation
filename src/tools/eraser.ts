import { segmentView } from "../widget";
import { AnnotationTool } from "./atool";

export class EraserTool extends AnnotationTool {
    toolIsActive : boolean;

    onStart(): void {
        // pass
    }
    
  mouseDown(e: MouseEvent, view: segmentView): void {

          view.previewLContext.clearRect(0, 0, view.previewLCanvas.width, view.previewLCanvas.height);  // clear preview, need to redraw for the ones that persist preview
    const [mouseX, mouseY] = view.canvasCoords(e);
    // loads annIContext data into preview canvas
    // view.previewLContext.clearRect(0, 0, view.previewLCanvas.width, view.previewLCanvas.height);
    view.previewLContext.globalAlpha = view.model.alpha * 1.15;  // needed because default is solid
    // view.previewLContext.fillStyle = `rgba(0, 255, 0, ${view.model.alpha * 1.15})`;
    // view.previewLContext.fillRect(0, 0, view.previewLCanvas.width, view.previewLCanvas.height);
    view.previewLContext.drawImage(
      view.model.annICanvas,
      view._Sx, // sx
      view._Sy, // sy
      view._sWidth,
      view._sHeight,
      0, // dx
      0, // dy
      view.displayLCanvas.width,
      view.displayLCanvas.height
    );
    view.previewLContext.globalAlpha = 1  // restore default
    view.model.annIContext.clearRect(view._Sx, // sx
      view._Sy, // sy
      view._sWidth,
      view._sHeight);  // clear class Canvas on the visible region
    view.redraw()  // update annIContext in display
    // view.drawImageScaled()  // update annIContext in display

    // start erasing
    this.toolIsActive = true;
    view.previewLContext.clearRect(mouseX - Math.floor(view.model.toolSize / 2),
      mouseY - Math.floor(view.model.toolSize / 2), view.model.toolSize, view.model.toolSize);
  }

mouseMove(e: MouseEvent, view: segmentView): void {
    if (this.toolIsActive) {
      const [mouseX, mouseY] = view.canvasCoords(e);
      view.previewLContext.clearRect(mouseX - Math.floor(view.model.toolSize / 2),
        mouseY - Math.floor(view.model.toolSize / 2), view.model.toolSize, view.model.toolSize);
    }
  }

mouseUp(e: MouseEvent, view: segmentView): void {
    // push current preview to class canvas
    this.toolIsActive = false;
    view.model.annIContext.drawImage(  // put preview on class canvas
      view.previewLCanvas,
      0,
      0,
      view.previewLCanvas.width,
      view.previewLCanvas.height,
      view._Sx,
      view._Sy,
      view._sWidth,
      view._sHeight
    );
    view.pushAnnToBackend()
    // view._drawInAnnAlias(view.previewLCanvas)
    // view.redraw()  // update annIContext in display
    view.previewLContext.clearRect(0, 0, view.previewLCanvas.width, view.previewLCanvas.height);  // clear preview
    // sync the data after all the drawing to keep drawing as snappy as possible
  }




}



