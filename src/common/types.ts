import * as tf from '@tensorflow/tfjs-core';

export type ConvParams = {
  filters: tf.Tensor4D
  bias: tf.Tensor1D
}

export type FCParams = {
  weights: tf.Tensor2D
  bias: tf.Tensor1D
}

export class SeparableConvParams {
  constructor(
    public depthwise_filter: tf.Tensor4D,
    public pointwise_filter: tf.Tensor4D,
    public bias: tf.Tensor1D
  ) {}
}