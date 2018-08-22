import * as tf from '@tensorflow/tfjs-core';

import { TinyYolov2LossFunction } from '../src/tinyYolov2/TinyYolov2LossFunction';
import { GroundTruth, GroundTruthWithGridPosition } from '../src/tinyYolov2/types';

export function expectTensorToBeZeros(tensor: tf.Tensor) {
  return expect(Array.from(tensor.dataSync()).reduce((sum, val) => sum + val, 0)).toEqual(0)
}

export function createFakeConfig(config: any = {}) {
  return {
    withSeparableConvs: true,
    iouThreshold: 0.4,
    anchors: [
      { x: 1, y: 2 }
    ],
    classes: ['foo', 'bar'],
    objectScale: 5,
    noObjectScale: 1,
    coordScale: 1,
    classScale: 0,
    ...config
  }
}

export function createFakeLossFunction(numCells: number, groundTruth: GroundTruth[], predictedBoxes: GroundTruthWithGridPosition[], config: any = {}) {
  return new TinyYolov2LossFunction(
    tf.ones([1, numCells, numCells, 7]) as tf.Tensor4D,
    groundTruth,
    predictedBoxes,
    { width: numCells * 32, height: numCells * 32 },
    createFakeConfig(config)
  )
}

export function createFakeLossFunctionMd(groundTruth: GroundTruth[], predictedBoxes: GroundTruthWithGridPosition[], config: any = {}) {
  return createFakeLossFunction(13, groundTruth, predictedBoxes, config)
}