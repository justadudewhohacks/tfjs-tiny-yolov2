import { ParamMapping } from 'tfjs-image-recognition-base';
import { TinyYolov2Config } from './config';
import { TinyYolov2NetParams } from './types';
export declare function extractParams(weights: Float32Array, config: TinyYolov2Config, boxEncodingSize: number, filterSizes: number[]): {
    params: TinyYolov2NetParams;
    paramMappings: ParamMapping[];
};
