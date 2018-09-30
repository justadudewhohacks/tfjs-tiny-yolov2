import * as tf from '@tensorflow/tfjs-core';
import { Dimensions } from 'tfjs-image-recognition-base';
import { TinyYolov2TrainableConfig } from './config';
import { TinyYolov2 } from './TinyYolov2';
import { GroundTruth, TinyYolov2BackwardOptions } from './types';
export declare class TinyYolov2Trainable extends TinyYolov2 {
    private _trainableConfig;
    private _optimizer;
    constructor(trainableConfig: TinyYolov2TrainableConfig, optimizer: tf.Optimizer);
    readonly trainableConfig: TinyYolov2TrainableConfig;
    readonly optimizer: tf.Optimizer;
    backward(img: HTMLImageElement | HTMLCanvasElement, groundTruth: GroundTruth[], inputSize: number, options?: TinyYolov2BackwardOptions): Promise<tf.Tensor<tf.Rank.R0> | null>;
    computeLoss(outputTensor: tf.Tensor4D, groundTruth: GroundTruth[], reshapedImgDims: Dimensions): {
        noObjectLoss: tf.Tensor<tf.Rank.R0>;
        objectLoss: tf.Tensor<tf.Rank.R0>;
        coordLoss: tf.Tensor<tf.Rank.R0>;
        classLoss: tf.Tensor<tf.Rank.R0>;
        totalLoss: tf.Tensor<tf.Rank.R0>;
    };
    filterGroundTruthBoxes(groundTruth: GroundTruth[], imgDims: Dimensions, minBoxSize: number): GroundTruth[];
    load(weightsOrUrl: Float32Array | string | undefined): Promise<void>;
}
