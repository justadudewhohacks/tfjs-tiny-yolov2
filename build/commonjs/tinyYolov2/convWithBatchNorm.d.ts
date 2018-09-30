import * as tf from '@tensorflow/tfjs-core';
import { SeparableConvParams } from '../common/types';
import { ConvWithBatchNorm } from './types';
export declare function convWithBatchNorm(x: tf.Tensor4D, params: ConvWithBatchNorm | SeparableConvParams): tf.Tensor4D;
