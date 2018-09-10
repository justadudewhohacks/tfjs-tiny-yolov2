import { ExtractWeightsFunction, ParamMapping } from 'tfjs-image-recognition-base';
import { ConvParams } from './types';
export declare function extractConvParamsFactory(extractWeights: ExtractWeightsFunction, paramMappings: ParamMapping[]): (channelsIn: number, channelsOut: number, filterSize: number, mappedPrefix: string) => ConvParams;
