import { ParamMapping } from 'tfjs-image-recognition-base';
import { NetParams } from './types';
export declare function loadQuantizedParams(uri: string, withSeparableConvs: boolean, defaultModelName?: string): Promise<{
    params: NetParams;
    paramMappings: ParamMapping[];
}>;
