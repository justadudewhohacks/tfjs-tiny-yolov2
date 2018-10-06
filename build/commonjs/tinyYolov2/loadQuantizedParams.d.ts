import { ParamMapping } from 'tfjs-image-recognition-base';
import { TinyYolov2Config } from './config';
import { NetParams } from './types';
export declare function loadQuantizedParams(uri: string, config: TinyYolov2Config, defaultModelName?: string): Promise<{
    params: NetParams;
    paramMappings: ParamMapping[];
}>;
