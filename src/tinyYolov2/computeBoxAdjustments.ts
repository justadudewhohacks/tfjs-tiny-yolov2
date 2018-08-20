import * as tf from '@tensorflow/tfjs-core';

import { Point } from '../Point';
import { Dimensions } from '../types';
import { CELL_SIZE, getNumCells } from './const';
import { GroundTruthWithGridPosition } from './types';


export function computeBoxAdjustments(groundTruthBoxes: GroundTruthWithGridPosition[], anchors: Point[], reshapedImgDims: Dimensions) {

  const inputSize = Math.max(reshapedImgDims.width, reshapedImgDims.height)
  const numCells = getNumCells(inputSize)

  const adjustments = tf.zeros([numCells, numCells, 25])
  const buf = adjustments.buffer()

  groundTruthBoxes.forEach(({ row, col, anchor, box }) => {
    const { left, top, right, bottom, width, height } = box.rescale(reshapedImgDims)

    const centerX = (left + right) / 2
    const centerY = (top + bottom) / 2

    const dCenterX = centerX - (col * CELL_SIZE)
    const dCenterY = centerY - (row * CELL_SIZE)

    const dx = dCenterX / CELL_SIZE
    const dy = dCenterY / CELL_SIZE
    const dw = Math.log((width / CELL_SIZE) / anchors[anchor].x)
    const dh = Math.log((height / CELL_SIZE) / anchors[anchor].y)

    const anchorOffset = anchor * 5
    buf.set(dx, row, col, anchorOffset + 0)
    buf.set(dy, row, col, anchorOffset + 1)
    buf.set(dw, row, col, anchorOffset + 2)
    buf.set(dh, row, col, anchorOffset + 3)
  })

  return adjustments
}

