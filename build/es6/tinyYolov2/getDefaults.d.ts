import { SizeType, TinyYolov2ForwardParams, TinyYolov2BackwardOptions } from './types';
export declare function getDefaultForwardParams(params: TinyYolov2ForwardParams): {
    inputSize: SizeType;
    scoreThreshold: number;
} & TinyYolov2ForwardParams;
export declare function getDefaultBackwardOptions(options: TinyYolov2BackwardOptions): {
    minBoxSize: number;
} & TinyYolov2BackwardOptions;
