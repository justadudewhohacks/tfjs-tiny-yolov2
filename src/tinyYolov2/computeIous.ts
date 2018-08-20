import * as tf from '@tensorflow/tfjs-core';

import { iou } from '../iou';
import { Dimensions } from '../types';
import { getNumCells } from './const';
import { GridPosition, GroundTruthWithGridPosition } from './types';

export function computeIous(predBoxes: GroundTruthWithGridPosition[], groundTruthBoxes: GroundTruthWithGridPosition[], reshapedImgDims: Dimensions) {

  const inputSize = Math.max(reshapedImgDims.width, reshapedImgDims.height)
  const numCells = getNumCells(inputSize)

  const isSameAnchor = (p1: GridPosition) => (p2: GridPosition) =>
    p1.row === p2.row
      && p1.col === p2.col
      && p1.anchor === p2.anchor

  const ious = tf.zeros([numCells, numCells, 25])
  const buf = ious.buffer()

  groundTruthBoxes.forEach(({ row, col, anchor, box }) => {
    const predBox = predBoxes.find(isSameAnchor({ row, col, anchor }))

    if (!predBox) {
      throw new Error(`no output box found for: row ${row}, col ${col}, anchor ${anchor}`)
    }

    const boxIou = iou(
      box.rescale(reshapedImgDims),
      predBox.box.rescale(reshapedImgDims)
    )

    const anchorOffset = anchor * 5
    buf.set(boxIou, row, col, anchorOffset + 4)
  })

  return ious
}


