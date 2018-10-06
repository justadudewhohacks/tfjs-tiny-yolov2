import { Point } from 'tfjs-image-recognition-base';
export declare type TinyYolov2Config = {
    withSeparableConvs: boolean;
    iouThreshold: number;
    anchors: Point[];
    classes: string[];
    meanRgb?: [number, number, number];
    withClassScores?: boolean;
    filterSizes?: number[];
    isFirstLayerConv2d?: boolean;
};
export declare type TinyYolov2TrainableConfig = TinyYolov2Config & {
    noObjectScale: number;
    objectScale: number;
    coordScale: number;
    classScale: number;
};
export declare function validateConfig(config: any): void;
export declare function validateTrainConfig(config: any): TinyYolov2TrainableConfig;
