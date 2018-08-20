import { Point } from '../Point';
export declare type TinyYolov2Config = {
    withSeparableConvs: boolean;
    iouThreshold: number;
    anchors: Point[];
    classes: string[];
    meanRgb?: [number, number, number];
    withClassScores?: boolean;
};
export declare function validateConfig(config: any): void;
