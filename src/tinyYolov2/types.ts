import * as tf from '@tensorflow/tfjs-core';

import { BoundingBox } from '../BoundingBox';
import { ConvParams } from '../commons/types';
import { IRect } from '../Rect';

export type BatchNorm = {
  sub: tf.Tensor1D
  truediv: tf.Tensor1D
}

export class SeparableConvParams {
  constructor(
    public depthwise_filter: tf.Tensor4D,
    public pointwise_filter: tf.Tensor4D,
    public bias: tf.Tensor1D
  ) {}
}

export type ConvWithBatchNorm = {
  conv: ConvParams
  bn: BatchNorm
}

export type NetParams = {
  conv0: ConvWithBatchNorm | SeparableConvParams
  conv1: ConvWithBatchNorm | SeparableConvParams
  conv2: ConvWithBatchNorm | SeparableConvParams
  conv3: ConvWithBatchNorm | SeparableConvParams
  conv4: ConvWithBatchNorm | SeparableConvParams
  conv5: ConvWithBatchNorm | SeparableConvParams
  conv6: ConvWithBatchNorm | SeparableConvParams
  conv7: ConvWithBatchNorm | SeparableConvParams
  conv8: ConvParams
}

export enum SizeType {
  XS = 'xs',
  SM = 'sm',
  MD = 'md',
  LG = 'lg'
}

export type TinyYolov2ForwardParams = {
  inputSize?: SizeType | number
  scoreThreshold?: number
}

export type GridPosition = {
  row: number
  col: number
  anchor: number
}

export type GroundTruthWithGridPosition = GridPosition & {
  box: BoundingBox
  classLabel: number
}

export type GroundTruth = IRect & {
  classLabel: number
}