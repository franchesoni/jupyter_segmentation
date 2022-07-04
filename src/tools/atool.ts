import { segmentView } from "../widget";

// define an annotation tool abstract class
export abstract class AnnotationTool {
    abstract toolIsActive: boolean;
    abstract onStart(view: segmentView): void;
    abstract mouseDown(event: MouseEvent, view: segmentView): void;
    abstract mouseMove(event: MouseEvent, view: segmentView): void;
    abstract mouseUp(event: MouseEvent, view:segmentView): void;
};