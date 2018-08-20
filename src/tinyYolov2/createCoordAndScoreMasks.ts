import * as tf from '@tensorflow/tfjs-core';

import { getNumCells } from './const';

export function createCoordAndScoreMasks(
  inputSize: number,
  boxEncodingSize: number,
  numAnchors: number
) {

  const numCells = getNumCells(inputSize)
  const gridCellEncodingSize = boxEncodingSize * numAnchors

  const coordMask = tf.zeros([numCells, numCells, gridCellEncodingSize])
  const scoreMask = tf.zeros([numCells, numCells, gridCellEncodingSize])
  const coordBuf = coordMask.buffer()
  const scoreBuf = scoreMask.buffer()

  for (let row = 0; row < numCells; row++) {
    for (let col = 0; col < numCells; col++) {
      for (let anchor = 0; anchor < numAnchors; anchor++) {
        const anchorOffset = boxEncodingSize * anchor
        for (let i = 0; i < 4; i++) {
          coordBuf.set(1, row, col, anchorOffset + i)
        }
        scoreBuf.set(1, row, col, anchorOffset + 4)
      }
    }
  }

  return { coordMask, scoreMask }
}