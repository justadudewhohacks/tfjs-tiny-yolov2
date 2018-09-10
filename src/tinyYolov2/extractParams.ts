import * as tf from '@tensorflow/tfjs-core';
import { extractWeightsFactory, ExtractWeightsFunction, ParamMapping } from 'tfjs-image-recognition-base';

import { extractConvParamsFactory } from '../common';
import { extractSeparableConvParamsFactory } from '../common/extractSeparableConvParamsFactory';
import { BatchNorm, ConvWithBatchNorm, NetParams } from './types';

function extractorsFactory(extractWeights: ExtractWeightsFunction, paramMappings: ParamMapping[]) {

  const extractConvParams = extractConvParamsFactory(extractWeights, paramMappings)

  function extractBatchNormParams(size: number, mappedPrefix: string): BatchNorm {

    const sub = tf.tensor1d(extractWeights(size))
    const truediv = tf.tensor1d(extractWeights(size))

    paramMappings.push(
      { paramPath: `${mappedPrefix}/sub` },
      { paramPath: `${mappedPrefix}/truediv` }
    )

    return { sub, truediv }
  }

  function extractConvWithBatchNormParams(channelsIn: number, channelsOut: number, mappedPrefix: string): ConvWithBatchNorm {

    const conv = extractConvParams(channelsIn, channelsOut, 3, `${mappedPrefix}/conv`)
    const bn = extractBatchNormParams(channelsOut, `${mappedPrefix}/bn`)

    return { conv, bn }
  }
  const extractSeparableConvParams = extractSeparableConvParamsFactory(extractWeights, paramMappings)

  return {
    extractConvParams,
    extractConvWithBatchNormParams,
    extractSeparableConvParams
  }

}

export function extractParams(
  weights: Float32Array,
  withSeparableConvs: boolean,
  boxEncodingSize: number,
  filterSizes: number[]
): { params: NetParams, paramMappings: ParamMapping[] } {

  const {
    extractWeights,
    getRemainingWeights
  } = extractWeightsFactory(weights)

  const paramMappings: ParamMapping[] = []

  const {
    extractConvParams,
    extractConvWithBatchNormParams,
    extractSeparableConvParams
  } = extractorsFactory(extractWeights, paramMappings)

  const extractConvFn = withSeparableConvs ? extractSeparableConvParams : extractConvWithBatchNormParams

  const [s0, s1, s2, s3, s4, s5, s6, s7, s8] = filterSizes
  const conv0 = extractConvFn(s0, s1, 'conv0',)
  const conv1 = extractConvFn(s1, s2, 'conv1')
  const conv2 = extractConvFn(s2, s3, 'conv2')
  const conv3 = extractConvFn(s3, s4, 'conv3')
  const conv4 = extractConvFn(s4, s5, 'conv4')
  const conv5 = extractConvFn(s5, s6, 'conv5')
  const conv6 = extractConvFn(s6, s7, 'conv6')
  const conv7 = extractConvFn(s7, s8, 'conv7')
  const conv8 = extractConvParams(s8, 5 * boxEncodingSize, 1, 'conv8')

  if (getRemainingWeights().length !== 0) {
    throw new Error(`weights remaing after extract: ${getRemainingWeights().length}`)
  }

  const params = { conv0, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8 }

  return { params, paramMappings }
}