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
export declare type MobilenetParams = {
    conv0: SeparableConvParams | ConvParams;
    conv1: SeparableConvParams;
    conv2: SeparableConvParams;
    conv3: SeparableConvParams;
    conv4: SeparableConvParams;
    conv5: SeparableConvParams;
    conv6?: SeparableConvParams;
    conv7?: SeparableConvParams;
    conv8: ConvParams;
};
export declare type DefaultTinyYolov2NetParams = {
    conv0: ConvWithBatchNorm;
    conv1: ConvWithBatchNorm;
    conv2: ConvWithBatchNorm;
    conv3: ConvWithBatchNorm;
    conv4: ConvWithBatchNorm;
    conv5: ConvWithBatchNorm;
    conv6: ConvWithBatchNorm;
    conv7: ConvWithBatchNorm;
    conv8: ConvParams;
};
export declare type TinyYolov2NetParams = DefaultTinyYolov2NetParams | MobilenetParams;
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
