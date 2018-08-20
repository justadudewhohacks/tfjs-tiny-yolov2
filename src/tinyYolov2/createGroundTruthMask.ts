import * as tf from '@tensorflow/tfjs-core';

import { getNumCells } from './const';
import { GroundTruthWithGridPosition } from './types';

export function createGroundTruthMask(
  groundTruthBoxes: GroundTruthWithGridPosition[],
  inputSize: number,
  boxEncodingSize: number,
  numAnchors: number
) {

  const numCells = getNumCells(inputSize)
  const gridCellEncodingSize = boxEncodingSize * numAnchors

  const mask = tf.zeros([numCells, numCells, gridCellEncodingSize])
  const buf = mask.buffer()

  groundTruthBoxes.forEach(({ row, col, anchor, classLabel }) => {
    const anchorOffset = boxEncodingSize * anchor
    for (let i = 0; i < numAnchors; i++) {
      buf.set(1, row, col, anchorOffset + i)
    }
  })

  return mask
}