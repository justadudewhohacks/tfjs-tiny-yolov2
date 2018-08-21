import * as tf from '@tensorflow/tfjs-core';

import { BoundingBox } from '../BoundingBox';
import { iou } from '../iou';
import { Point } from '../Point';
import { Rect } from '../Rect';
import { Dimensions } from '../types';
import { TinyYolov2TrainableConfig } from './config';
import { CELL_SIZE } from './const';
import { GridPosition, GroundTruth, GroundTruthWithGridPosition } from './types';

export class TinyYolov2LossFunction {

  private _config: TinyYolov2TrainableConfig
  private _reshapedImgDims: Dimensions
  private _outputTensor: tf.Tensor4D
  private _groundTruth: GroundTruthWithGridPosition[]
  private _predictedBoxes: GroundTruthWithGridPosition[]

  private noObjectLossMask: tf.Tensor4D
  private objectLossMask: tf.Tensor4D
  private coordLossMask: tf.Tensor4D
  private groundTruthClassScoresMask: tf.Tensor4D

  constructor(
    outputTensor: tf.Tensor4D,
    groundTruth: GroundTruth[],
    predictedBoxes: GroundTruthWithGridPosition[],
    reshapedImgDims: Dimensions,
    config: TinyYolov2TrainableConfig
  ) {
    this._config = config
    this._reshapedImgDims = reshapedImgDims
    this._outputTensor = outputTensor
    this._groundTruth = this.assignGroundTruthToAnchors(groundTruth)
    this._predictedBoxes = predictedBoxes

    const groundTruthMask = this.createGroundTruthMask()
    const { coordMask, scoreMask } = this.createCoordAndScoreMasks()

    this.noObjectLossMask = tf.tidy(() => tf.mul(scoreMask, tf.sub(tf.scalar(1), groundTruthMask))) as tf.Tensor4D
    this.objectLossMask = tf.tidy(() => tf.mul(scoreMask, groundTruthMask)) as tf.Tensor4D
    this.coordLossMask = tf.tidy(() => tf.mul(coordMask, groundTruthMask)) as tf.Tensor4D
    this.groundTruthClassScoresMask = tf.tidy(() => tf.mul(tf.sub(tf.scalar(1), tf.add(coordMask, scoreMask)), groundTruthMask)) as tf.Tensor4D
  }

  public get config(): TinyYolov2TrainableConfig {
    return this._config
  }

  public get reshapedImgDims(): Dimensions {
    return this._reshapedImgDims
  }

  public get outputTensor(): tf.Tensor4D {
    return this._outputTensor
  }

  public get groundTruth(): GroundTruthWithGridPosition[] {
    return this._groundTruth
  }

  public get predictedBoxes(): GroundTruthWithGridPosition[] {
    return this._predictedBoxes
  }

  public get inputSize(): number {
    return Math.max(this.reshapedImgDims.width, this.reshapedImgDims.height)
  }

  public get withClassScores(): boolean {
    return this._config.withClassScores || this._config.classes.length > 1
  }

  public get boxEncodingSize(): number {
    return 5 + (this.withClassScores ? this._config.classes.length : 0)
  }

  public get anchors(): Point[] {
    return this._config.anchors
  }

  public get numBoxes(): number {
    return this.anchors.length
  }

  public get numCells(): number {
    return this.inputSize / CELL_SIZE
  }

  public get gridCellEncodingSize(): number {
    return this.boxEncodingSize * this.numBoxes
  }

  public computeLoss() {
    return tf.tidy(() => {

      const noObjectLoss = this.computeNoObjectLoss()
      const objectLoss = this.computeObjectLoss()
      const coordLoss = this.computeCoordLoss()
      const classLoss = this.withClassScores
        ? this.computeClassLoss()
        : tf.scalar(0)

      const totalLoss = tf.tidy(() => noObjectLoss.add(objectLoss).add(coordLoss).add(classLoss))

      return {
        noObjectLoss,
        objectLoss,
        coordLoss,
        classLoss,
        totalLoss
      }
    })
  }

  public computeNoObjectLoss(): tf.Tensor4D {
    return tf.tidy(() =>
      this.computeLossTerm(
        this.config.noObjectScale,
        this.noObjectLossMask,
        tf.sigmoid(this.outputTensor)
      )
    )
  }

  public computeObjectLoss(): tf.Tensor4D {
    return tf.tidy(() =>
      this.computeLossTerm(
        this.config.objectScale,
        this.objectLossMask,
        tf.sub(this.computeIous(), tf.sigmoid(this.outputTensor))
      )
    )
  }

  public computeCoordLoss(): tf.Tensor4D {
    return tf.tidy(() =>
      this.computeLossTerm(
        this.config.coordScale,
        this.coordLossMask,
        tf.sub(this.computeBoxAdjustments(), this.outputTensor)
      )
    )
  }

  public computeClassLoss(): tf.Tensor4D {
    const classLossTensor = tf.tidy(() => {
      const groundTruthClassValues = tf.mul(this.outputTensor, this.groundTruthClassScoresMask)
      //const reshaped = groundTruthClassValues.reshape([this.numCells, this.numCells, numBoxes, this.boxEncodingSize])

      // TBD
      const classScores = tf.softmax(groundTruthClassValues, 3)
      const gtClassScores = this.createOneHotClassScoreMask()

      return tf.sub(gtClassScores, classScores)
    })

    return tf.tidy(() =>
      this.computeLossTerm(
        this.config.classScale,
        tf.scalar(1),
        classLossTensor as tf.Tensor4D
      )
    )
  }

  private computeLossTerm(scale: number, mask: tf.Tensor<tf.Rank>, lossTensor: tf.Tensor4D): tf.Tensor4D {
    return tf.tidy(() => tf.mul(tf.scalar(scale), this.squaredSumOverMask(mask, lossTensor)))
  }

  private squaredSumOverMask(mask: tf.Tensor<tf.Rank>, lossTensor: tf.Tensor4D): tf.Tensor4D {
    return tf.tidy(() => tf.sum(tf.square(tf.mul(mask, lossTensor))))
  }

  private assignGroundTruthToAnchors(groundTruth: GroundTruth[]) {

    const groundTruthBoxes = groundTruth
      .map(({ x, y, width, height, classLabel }) => ({
        box: (new Rect(x, y, width, height)).toBoundingBox(),
        classLabel
      }))

    return groundTruthBoxes.map(({ box, classLabel }) => {
      const { left, top, width, height } = box.rescale(this.reshapedImgDims)

      const ctX = left + (width / 2)
      const ctY = top + (height / 2)

      const col = Math.floor((ctX / this.inputSize) * this.numCells)
      const row = Math.floor((ctY / this.inputSize) * this.numCells)

      const anchorsByIou = this.anchors.map((anchor, idx) => ({
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

  private createGroundTruthMask() {

    const gridCellEncodingSize = this.boxEncodingSize * this.numBoxes

    const mask = tf.zeros([this.numCells, this.numCells, gridCellEncodingSize])
    const buf = mask.buffer()

    this.groundTruth.forEach(({ row, col, anchor }) => {
      const anchorOffset = this.boxEncodingSize * anchor
      for (let i = 0; i < this.boxEncodingSize; i++) {
        buf.set(1, row, col, anchorOffset + i)
      }
    })

    return mask
  }

  private createCoordAndScoreMasks() {

    const gridCellEncodingSize = this.boxEncodingSize * this.numBoxes

    const coordMask = tf.zeros([this.numCells, this.numCells, gridCellEncodingSize])
    const scoreMask = tf.zeros([this.numCells, this.numCells, gridCellEncodingSize])
    const coordBuf = coordMask.buffer()
    const scoreBuf = scoreMask.buffer()

    for (let row = 0; row < this.numCells; row++) {
      for (let col = 0; col < this.numCells; col++) {
        for (let anchor = 0; anchor < this.numBoxes; anchor++) {
          const anchorOffset = this.boxEncodingSize * anchor
          for (let i = 0; i < 4; i++) {
            coordBuf.set(1, row, col, anchorOffset + i)
          }
          scoreBuf.set(1, row, col, anchorOffset + 4)
        }
      }
    }

    return { coordMask, scoreMask }
  }

  private createOneHotClassScoreMask() {
    const mask = tf.zeros([this.numCells, this.numCells, this.numBoxes * this.boxEncodingSize])
    const buf = mask.buffer()

    const classValuesOffset = 6
    this.groundTruth.forEach(({ row, col, anchor, classLabel }) => {
      const anchorOffset = this.boxEncodingSize * anchor
      buf.set(1, row, col, anchorOffset + classValuesOffset + classLabel)
    })

    return mask
  }

  private computeIous() {

    const isSameAnchor = (p1: GridPosition) => (p2: GridPosition) =>
      p1.row === p2.row
        && p1.col === p2.col
        && p1.anchor === p2.anchor

    const ious = tf.zeros([this.numCells, this.numCells, this.gridCellEncodingSize])
    const buf = ious.buffer()

    this.groundTruth.forEach(({ row, col, anchor, box }) => {
      const predBox = this.predictedBoxes.find(isSameAnchor({ row, col, anchor }))

      if (!predBox) {
        throw new Error(`no output box found for: row ${row}, col ${col}, anchor ${anchor}`)
      }

      const boxIou = iou(
        box.rescale(this.reshapedImgDims),
        predBox.box.rescale(this.reshapedImgDims)
      )

      const anchorOffset = anchor * 5
      const scoreValueOffset = 4
      buf.set(boxIou, row, col, anchorOffset + scoreValueOffset)
    })

    return ious
  }

  private computeBoxAdjustments() {

    const adjustments = tf.zeros([this.numCells, this.numCells, this.gridCellEncodingSize])
    const buf = adjustments.buffer()

    this.groundTruth.forEach(({ row, col, anchor, box }) => {
      const { left, top, right, bottom, width, height } = box.rescale(this.reshapedImgDims)

      const centerX = (left + right) / 2
      const centerY = (top + bottom) / 2

      const dCenterX = centerX - (col * CELL_SIZE)
      const dCenterY = centerY - (row * CELL_SIZE)

      const dx = dCenterX / CELL_SIZE
      const dy = dCenterY / CELL_SIZE
      const dw = Math.log((width / CELL_SIZE) / this.anchors[anchor].x)
      const dh = Math.log((height / CELL_SIZE) / this.anchors[anchor].y)

      const anchorOffset = anchor * 5
      buf.set(dx, row, col, anchorOffset + 0)
      buf.set(dy, row, col, anchorOffset + 1)
      buf.set(dw, row, col, anchorOffset + 2)
      buf.set(dh, row, col, anchorOffset + 3)
    })

    return adjustments
  }

}