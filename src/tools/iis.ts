import { segmentView } from "../widget";
import { AnnotationTool } from "./atool";

export class IISTool extends AnnotationTool {
    toolIsActive: boolean;

    onStart(): void {
        // pass
    }

    mouseDown(e: MouseEvent, view: segmentView): void {
        console.log('mouseDownIIS');
        const [mouseX, mouseY] = view.canvasCoords(e);
        // check if click is outside canvas (it's not working well)
        if ((mouseX < 0 || mouseX > view.previewLCanvas.width || mouseY < 0 || mouseY > view.previewLCanvas.height)) {
            console.log('click outside canvas');
        } else {
            if (e.button === 0) {  // left click = positive click
                // append click position to list
                view.model.pcs.add([mouseX, mouseY]);
                // view.model.pcs = [...view.model.pcs, [mouseX, mouseY]];
            }
            else if (e.button === 2) {  // right click = negative click
                // append click position to list
                view.model.ncs.add([mouseX, mouseY]);
            }
            view.previewLContext.clearRect(0, 0, view.previewLCanvas.width, view.previewLCanvas.height);
            view._drawClicks();
            view.model.set('pcs', Array.from(view.model.pcs))
            view.model.set('ncs', Array.from(view.model.ncs))
            view.model.save_changes()
        }


    }

    mouseMove(e: MouseEvent, view: segmentView): void {
        //pass
    }

    mouseUp(e: MouseEvent, view: segmentView): void {
        //pass
    }




}



