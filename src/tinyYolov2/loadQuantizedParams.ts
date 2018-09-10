import * as tf from '@tensorflow/tfjs-core';
import {
  disposeUnusedWeightTensors,
  extractWeightEntryFactory,
  loadWeightMap,
  ParamMapping,
} from 'tfjs-image-recognition-base';

import { ConvParams } from '../common';
import { loadSeparableConvParamsFactory } from '../common/extractSeparableConvParamsFactory';
import { BatchNorm, ConvWithBatchNorm, NetParams } from './types';

function extractorsFactory(weightMap: any, paramMappings: ParamMapping[]) {

  const extractWeightEntry = extractWeightEntryFactory(weightMap, paramMappings)

  function extractBatchNormParams(prefix: string): BatchNorm {
    const sub = extractWeightEntry<tf.Tensor1D>(`${prefix}/sub`, 1)
    const truediv = extractWeightEntry<tf.Tensor1D>(`${prefix}/truediv`, 1)
    return { sub, truediv }
  }

  function extractConvParams(prefix: string): ConvParams {
    const filters = extractWeightEntry<tf.Tensor4D>(`${prefix}/filters`, 4)
    const bias = extractWeightEntry<tf.Tensor1D>(`${prefix}/bias`, 1)
    return { filters, bias }
  }

  function extractConvWithBatchNormParams(prefix: string): ConvWithBatchNorm {
    const conv = extractConvParams(`${prefix}/conv`)
    const bn = extractBatchNormParams(`${prefix}/bn`)
    return { conv, bn }
  }

  const extractSeparableConvParams = loadSeparableConvParamsFactory(extractWeightEntry)

  return {
    extractConvParams,
    extractConvWithBatchNormParams,
    extractSeparableConvParams
  }

}

export async function loadQuantizedParams(
  uri: string,
  withSeparableConvs: boolean,
  defaultModelName: string = ''
): Promise<{ params: NetParams, paramMappings: ParamMapping[] }> {

  const weightMap = await loadWeightMap(uri, defaultModelName)
  const paramMappings: ParamMapping[] = []

  const {
    extractConvParams,
    extractConvWithBatchNormParams,
    extractSeparableConvParams
  } = extractorsFactory(weightMap, paramMappings)

  const extractConvFn = withSeparableConvs ? extractSeparableConvParams : extractConvWithBatchNormParams

  const params = {
    conv0: extractConvFn('conv0'),
    conv1: extractConvFn('conv1'),
    conv2: extractConvFn('conv2'),
    conv3: extractConvFn('conv3'),
    conv4: extractConvFn('conv4'),
    conv5: extractConvFn('conv5'),
    conv6: extractConvFn('conv6'),
    conv7: extractConvFn('conv7'),
    conv8: extractConvParams('conv8')
  }

  disposeUnusedWeightTensors(weightMap, paramMappings)

  return { params, paramMappings }
}