import { ObjectDetection } from '../ObjectDetection';
import { DrawBoxOptions, DrawDetectionOptions, DrawOptions, DrawTextOptions } from './types';
export declare function getDefaultDrawOptions(options?: any): DrawOptions;
export declare function drawBox(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, options: DrawBoxOptions): void;
export declare function drawText(ctx: CanvasRenderingContext2D, x: number, y: number, text: string, options: DrawTextOptions): void;
export declare function drawDetection(canvasArg: string | HTMLCanvasElement, detection: ObjectDetection | ObjectDetection[], options?: DrawDetectionOptions): void;
