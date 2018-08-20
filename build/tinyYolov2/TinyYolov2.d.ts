import * as tf from '@tensorflow/tfjs-core';
import { BoundingBox } from '../BoundingBox';
import { NeuralNetwork } from '../commons/NeuralNetwork';
import { NetInput } from '../NetInput';
import { ObjectDetection } from '../ObjectDetection';
import { Dimensions, TNetInput } from '../types';
import { TinyYolov2Config } from './config';
import { NetParams, TinyYolov2ForwardParams } from './types';
export declare class TinyYolov2 extends NeuralNetwork<NetParams> {
    private _config;
    constructor(config: TinyYolov2Config);
    readonly config: TinyYolov2Config;
    readonly withClassScores: boolean;
    readonly boxEncodingSize: number;
    forwardInput(input: NetInput, inputSize: number): tf.Tensor4D;
    forward(input: TNetInput, inputSize: number): Promise<tf.Tensor4D>;
    detect(input: TNetInput, forwardParams?: TinyYolov2ForwardParams): Promise<ObjectDetection[]>;
    extractBoxes(outputTensor: tf.Tensor4D, scoreThreshold: number, inputBlobDimensions: Dimensions): {
        row: number;
        col: number;
        anchor: number;
        box: BoundingBox;
        score: number;
        className: string;
    }[];
    extractClassScores(classesTensor: tf.Tensor4D, score: number, pos: {
        row: number;
        col: number;
        anchor: number;
    }): number[];
    protected loadQuantizedParams(modelUri: string | undefined): Promise<{
        params: NetParams;
        paramMappings: import("src/commons/types").ParamMapping[];
    }>;
    protected extractParams(weights: Float32Array): {
        params: NetParams;
        paramMappings: import("src/commons/types").ParamMapping[];
    };
}
