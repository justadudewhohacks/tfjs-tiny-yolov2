import * as tf from '@tensorflow/tfjs-core';
import { Box, IRect } from 'tfjs-image-recognition-base';
import { ConvParams } from '../common';
import { SeparableConvParams } from '../common/types';
export declare type BatchNorm = {
    sub: tf.Tensor1D;
    truediv: tf.Tensor1D;
};
export declare type ConvWithBatchNorm = {
    conv: ConvParams;
    bn: BatchNorm;
};
export declare type NetParams = {
    conv0: ConvWithBatchNorm | SeparableConvParams;
    conv1: ConvWithBatchNorm | SeparableConvParams;
    conv2: ConvWithBatchNorm | SeparableConvParams;
    conv3: ConvWithBatchNorm | SeparableConvParams;
    conv4: ConvWithBatchNorm | SeparableConvParams;
    conv5: ConvWithBatchNorm | SeparableConvParams;
    conv6: ConvWithBatchNorm | SeparableConvParams;
    conv7: ConvWithBatchNorm | SeparableConvParams;
    conv8: ConvParams;
};
export declare enum SizeType {
    XS = "xs",
    SM = "sm",
    MD = "md",
    LG = "lg",
}
export declare type GridPosition = {
    row: number;
    col: number;
    anchor: number;
};
export declare type GroundTruthWithGridPosition = GridPosition & {
    box: Box;
    label: number;
};
export declare type GroundTruth = IRect & {
    label: number;
};
export declare type TinyYolov2ForwardParams = {
    inputSize?: SizeType | number;
    scoreThreshold?: number;
};
export declare type YoloLoss = {
    totalLoss: number;
    noObjectLoss: number;
    objectLoss: number;
    coordLoss: number;
    classLoss: number;
};
export declare type LossReport = {
    losses: YoloLoss;
    numBoxes: number;
    inputSize: number;
};
export declare type TinyYolov2BackwardOptions = {
    minBoxSize?: number;
    reportLosses?: (report: LossReport) => void;
};
