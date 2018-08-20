import * as tf from '@tensorflow/tfjs-core';

import { getNumCells } from './const';
import { GroundTruthWithGridPosition } from './types';

export function createOneHotClassScoreMask(
  groundTruthBoxes: GroundTruthWithGridPosition[],
  numCells: number,
  numClasses: number,
  numAnchors: number
) {
  const mask = tf.zeros([numCells, numCells, numAnchors, numClasses])
  const buf = mask.buffer()

  groundTruthBoxes.forEach(({ row, col, anchor, classLabel }) => {
    buf.set(1, row, col, anchor, classLabel)
  })

  return mask
}