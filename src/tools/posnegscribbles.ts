import { segmentView } from "../widget";
import { AnnotationTool } from "./atool";

export class PosNegScribbles extends AnnotationTool {
    toolIsActive: boolean;
    toolIsActiveL: boolean;
    toolIsActiveR: boolean;

    onStart(): void {
        // pass
    }

    mouseDown(e: MouseEvent, view: segmentView): void {
        if (view.model.tool !== 7  || e.button !== 2 || view.model.pcs.size > view.model.ncs.size + 1){
        
        console.log('mouseDownScribbles');
        const [mouseX, mouseY] = view.canvasCoords(e);
        if (e.button === 0) {  // left click = positive click
            this.toolIsActiveL = true;
            // append click position to list
            view.model.pcs.add([mouseX, mouseY]);
            view.model.ncs.delete([mouseX, mouseY]);
        }
        else if (e.button === 2) {  // right click = negative click
            this.toolIsActiveR = true;
            // append click position to list
            view.model.ncs.add([mouseX, mouseY]);
            view.model.pcs.delete([mouseX, mouseY]);
        }
        this.toolIsActive = this.toolIsActiveL || this.toolIsActiveR;
        view._drawClicks();
        view.model.set('pcs', Array.from(view.model.pcs))
        view.model.set('ncs', Array.from(view.model.ncs))
        view.model.save_changes()
    }
}

    mouseMove(e: MouseEvent, view: segmentView): void {
        const [mouseX, mouseY] = view.canvasCoords(e);
        if (this.toolIsActiveL) {
            // append click position to list
            view.model.pcs.add([mouseX, mouseY]);
            view.model.ncs.delete([mouseX, mouseY]);
        }
        else if (this.toolIsActiveR) {  // right click = negative click
            // append click position to list
            view.model.ncs.add([mouseX, mouseY]);
            view.model.pcs.delete([mouseX, mouseY]);
        }
        // view.previewLContext.clearRect(0, 0, view.previewLCanvas.width, view.previewLCanvas.height);
        view._drawClicks();
        view.model.set('pcs', Array.from(view.model.pcs))
        view.model.set('ncs', Array.from(view.model.ncs))
        view.model.save_changes()
    }

    mouseUp(e: MouseEvent, view: segmentView): void {
        if (e.button === 0) {
            this.toolIsActiveL = false;
        }
        else if (e.button === 2) {
            this.toolIsActiveR = false;
        }

        this.toolIsActive = this.toolIsActiveL || this.toolIsActiveR;
    }

}