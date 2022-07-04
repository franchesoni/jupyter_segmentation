import { segmentView } from "../widget";
import { AnnotationTool } from "./atool";

export class SuperpixTool extends AnnotationTool {
    toolIsActive: boolean;

    onStart(): void {
        // pass
    }



    mouseDown(e: MouseEvent, view: segmentView): void {
    console.log('mouseDownSuperpix');
    this.toolIsActive = true;
    const [mouseX, mouseY] = view.canvasCoords(e);
    if (e.button === 0) {  // left click = positive click
      // append click position to list
      view.model.pcs = [...view.model.pcs, [mouseX, mouseY]];
    }

    // view.previewLContext.clearRect(0, 0, view.previewLCanvas.width, view.previewLCanvas.height);
    view._drawClicks();
    view.model.set('pcs', view.model.pcs)
    view.model.save_changes()

    }

    mouseMove(e: MouseEvent, view: segmentView): void {
    const [mouseX, mouseY] = view.canvasCoords(e);
    if (e.button === 0 && this.toolIsActive) {  // left click = positive click
      // append click position to list
      view.model.pcs = [...view.model.pcs, [mouseX, mouseY]];
    }
    // view.previewLContext.clearRect(0, 0, view.previewLCanvas.width, view.previewLCanvas.height);
    view._drawClicks();
    view.model.set('pcs', view.model.pcs)
    view.model.save_changes()

    }

    mouseUp(e: MouseEvent, view: segmentView): void {
    this.toolIsActive = false;
    }




}
