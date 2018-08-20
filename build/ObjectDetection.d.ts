import { Rect } from './Rect';
import { Dimensions } from './types';
export declare class ObjectDetection {
    private _score;
    private _className;
    private _box;
    private _imageWidth;
    private _imageHeight;
    constructor(score: number, className: string, relativeBox: Rect, imageDims: Dimensions);
    readonly score: number;
    readonly className: string;
    readonly box: Rect;
    readonly imageWidth: number;
    readonly imageHeight: number;
    readonly relativeBox: Rect;
    getScore(): number;
    getBox(): Rect;
    getImageWidth(): number;
    getImageHeight(): number;
    getRelativeBox(): Rect;
    forSize(width: number, height: number): ObjectDetection;
}
