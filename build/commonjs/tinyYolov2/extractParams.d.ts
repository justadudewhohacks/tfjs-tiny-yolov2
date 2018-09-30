import { ParamMapping } from 'tfjs-image-recognition-base';
import { NetParams } from './types';
export declare function extractParams(weights: Float32Array, withSeparableConvs: boolean, boxEncodingSize: number, filterSizes: number[]): {
    params: NetParams;
    paramMappings: ParamMapping[];
};
