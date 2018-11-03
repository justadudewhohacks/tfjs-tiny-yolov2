import * as tf from '@tensorflow/tfjs-core';
import { BoundingBox, Dimensions, NetInput, NeuralNetwork, ObjectDetection, TNetInput } from 'tfjs-image-recognition-base';
import { TinyYolov2Config } from './config';
import { ITinyYolov2Options } from './TinyYolov2Options';
import { DefaultTinyYolov2NetParams, MobilenetParams, TinyYolov2NetParams } from './types';
export declare class TinyYolov2 extends NeuralNetwork<TinyYolov2NetParams> {
    private _config;
    constructor(config: TinyYolov2Config);
    readonly config: TinyYolov2Config;
    readonly withClassScores: boolean;
    readonly boxEncodingSize: number;
    runTinyYolov2(x: tf.Tensor4D, params: DefaultTinyYolov2NetParams): tf.Tensor4D;
    runMobilenet(x: tf.Tensor4D, params: MobilenetParams): tf.Tensor4D;
    forwardInput(input: NetInput, inputSize: number): tf.Tensor4D;
    forward(input: TNetInput, inputSize: number): Promise<tf.Tensor4D>;
    detect(input: TNetInput, forwardParams?: ITinyYolov2Options): Promise<ObjectDetection[]>;
    protected getDefaultModelName(): string;
    protected extractParamsFromWeigthMap(weightMap: tf.NamedTensorMap): {
        params: TinyYolov2NetParams;
        paramMappings: {
            originalPath?: string | undefined;
            paramPath: string;
        }[];
    };
    protected extractParams(weights: Float32Array): {
        params: TinyYolov2NetParams;
        paramMappings: {
            originalPath?: string | undefined;
            paramPath: string;
        }[];
    };
    protected extractBoxes(outputTensor: tf.Tensor4D, inputBlobDimensions: Dimensions, scoreThreshold?: number): {
        row: number;
        col: number;
        anchor: number;
        box: BoundingBox;
        score: number;
        classScore: number;
        label: number;
    }[];
    private extractPredictedClass(classesTensor, pos);
}
