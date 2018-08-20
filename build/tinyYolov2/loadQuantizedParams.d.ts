import { ParamMapping } from '../commons/types';
import { NetParams } from './types';
export declare function loadQuantizedParams(uri: string, withSeparableConvs: boolean): Promise<{
    params: NetParams;
    paramMappings: ParamMapping[];
}>;
