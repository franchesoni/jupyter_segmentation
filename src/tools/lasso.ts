import { segmentView } from "../widget";
import { AnnotationTool } from "./atool";

export class LassoTool extends AnnotationTool {
    path : Path2D;
    toolIsActive : boolean;

    onStart(): void {
        // pass
    }
    mouseDown(e: MouseEvent, view:segmentView): void {

          view.previewLContext.clearRect(0, 0, view.previewLCanvas.width, view.previewLCanvas.height);  // clear preview, need to redraw for the ones that persist preview
        const [mouseX, mouseY] = view.canvasCoords(e);  // access a function
        this.path = new Path2D();
        this.path.moveTo(mouseX, mouseY);
        this.toolIsActive = true;
  }

mouseMove(e: MouseEvent, view:segmentView): void {
    if (this.toolIsActive) {
      const [mouseX, mouseY] = view.canvasCoords(e);
      view.previewLContext.clearRect(0, 0, view.previewLCanvas.width, view.previewLCanvas.height);
      this.path.lineTo(mouseX, mouseY);
      const closedPath = new Path2D(this.path);
      closedPath.closePath();
      view.previewLContext.fillStyle = `rgba(255, 0, 255, ${view.model.alpha * 1.15})`;
      view.previewLContext.fill(closedPath);
      view.previewLContext.setLineDash([15, 5]);
      view.previewLContext.stroke(this.path);
    }
  }

  mouseUp= (e: MouseEvent, view:segmentView): void => {
    if (this.toolIsActive) {
      view.previewLContext.clearRect(0, 0, view.previewLCanvas.width, view.previewLCanvas.height);
      view.previewLContext.fillStyle = `rgb(255, 0, 255)`;  // no alpha, it is given in drawimage
      view.previewLContext.fill(this.path);
      view._drawInAnnAlias(view.previewLCanvas)

      view.previewLContext.clearRect(0, 0, view.previewLCanvas.width, view.previewLCanvas.height);
      this.toolIsActive = false;

      // sync the data after all the drawing to keep drawing as snappy as possible
      view.pushAnnToBackend()
    }
  }


}

