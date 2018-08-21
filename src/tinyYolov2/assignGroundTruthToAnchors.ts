import { BoundingBox } from '../BoundingBox';
import { iou } from '../iou';
import { Point } from '../Point';
import { Rect } from '../Rect';
import { Dimensions } from '../types';
import { CELL_SIZE, getNumCells } from './const';
import { GroundTruth } from './types';

export function assignGroundTruthToAnchors(groundTruth: GroundTruth[], anchors: Point[], reshapedImgDims: Dimensions) {

  const inputSize = Math.max(reshapedImgDims.width, reshapedImgDims.height)
  const numCells = getNumCells(inputSize)

  const groundTruthBoxes = groundTruth
    .map(({ x, y, width, height, classLabel }) => ({
      box: (new Rect(x, y, width, height)).toBoundingBox(),
      classLabel
    }))

  return groundTruthBoxes.map(({ box, classLabel }) => {
    const { left, top, width, height } = box.rescale(reshapedImgDims)

    const ctX = left + (width / 2)
    const ctY = top + (height / 2)

    const col = Math.floor((ctX / inputSize) * numCells)
    const row = Math.floor((ctY / inputSize) * numCells)

    const anchorsByIou = anchors.map((anchor, idx) => ({
      idx,
      iou: iou(
        new BoundingBox(0, 0, anchor.x * CELL_SIZE, anchor.y * CELL_SIZE),
        new BoundingBox(0, 0, width, height)
      )
    })).sort((a1, a2) => a2.iou - a1.iou)

    const anchor = anchorsByIou[0].idx

    return { row, col, anchor, box, classLabel }
  })
}