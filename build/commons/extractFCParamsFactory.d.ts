import { ExtractWeightsFunction, ParamMapping } from 'tfjs-image-recognition-base';
import { FCParams } from './types';
export declare function extractFCParamsFactory(extractWeights: ExtractWeightsFunction, paramMappings: ParamMapping[]): (channelsIn: number, channelsOut: number, mappedPrefix: string) => FCParams;
