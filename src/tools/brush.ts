import { segmentView } from "../widget";
import { AnnotationTool } from "./atool";

export class BrushTool extends AnnotationTool {
    toolIsActive : boolean;

    onStart(): void {
        // pass
    }
    
  mouseDown(e: MouseEvent, view: segmentView): void {
    // clears and edits preview
          view.previewLContext.clearRect(0, 0, view.previewLCanvas.width, view.previewLCanvas.height);  // clear preview, need to redraw for the ones that persist preview
    const [mouseX, mouseY] = view.canvasCoords(e);
    view.previewLContext.clearRect(0, 0, view.previewLCanvas.width, view.previewLCanvas.height);
    view.previewLContext.fillStyle = `rgba(255, 0, 255, ${view.model.alpha * 1.15})`;
    view.previewLContext.fillRect(mouseX - Math.floor(view.model.toolSize / 2),
      mouseY - Math.floor(view.model.toolSize / 2), view.model.toolSize, view.model.toolSize);
    this.toolIsActive = true;
    console.log('mouse down brush')
  }

  mouseMove(e: MouseEvent, view: segmentView): void {
    // adds to preview
    if (this.toolIsActive) {
      const [mouseX, mouseY] = view.canvasCoords(e);
      view.previewLContext.fillStyle = `rgba(255, 0, 255, ${view.model.alpha * 1.15})`;
      view.previewLContext.fillRect(mouseX - Math.floor(view.model.toolSize / 2),
        mouseY - Math.floor(view.model.toolSize / 2), view.model.toolSize, view.model.toolSize);
    }
  }



  mouseUp(e: MouseEvent, view: segmentView): void {
    // writes preview to ann, clears it, pushes ann to backend
    this.toolIsActive = false;
    view._drawInAnnAlias(view.previewLCanvas)
    view.previewLContext.clearRect(0, 0, view.previewLCanvas.width, view.previewLCanvas.height);  // clear preview
    // sync the data after all the drawing to keep drawing as snappy as possible
    view.pushAnnToBackend()
  }



}

