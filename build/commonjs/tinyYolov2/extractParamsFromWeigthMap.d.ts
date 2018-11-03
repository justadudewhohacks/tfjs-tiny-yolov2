import * as tf from '@tensorflow/tfjs-core';
import { ParamMapping } from 'tfjs-image-recognition-base';
import { TinyYolov2Config } from './config';
import { TinyYolov2NetParams } from './types';
export declare function extractParamsFromWeigthMap(weightMap: tf.NamedTensorMap, config: TinyYolov2Config): {
    params: TinyYolov2NetParams;
    paramMappings: ParamMapping[];
};
