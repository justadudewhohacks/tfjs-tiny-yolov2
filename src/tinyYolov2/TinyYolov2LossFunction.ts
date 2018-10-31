import * as tf from '@tensorflow/tfjs-core';
import { BoundingBox, Dimensions, IDimensions, iou, Point, Rect } from 'tfjs-image-recognition-base';

import { TinyYolov2TrainableConfig } from './config';
import { CELL_SIZE } from './const';
import { GridPosition, GroundTruth, GroundTruthWithGridPosition } from './types';

export class TinyYolov2LossFunction {

  private _config: TinyYolov2TrainableConfig
  private _reshapedImgDims: Dimensions
  private _outputTensor: tf.Tensor4D
  private _groundTruth: GroundTruthWithGridPosition[]
  private _predictedBoxes: GroundTruthWithGridPosition[]

  public noObjectLossMask: tf.Tensor4D
  public objectLossMask: tf.Tensor4D
  public coordBoxOffsetMask: tf.Tensor4D
  public coordBoxSizeMask: tf.Tensor4D
  public groundTruthClassScoresMask: tf.Tensor4D

  constructor(
    outputTensor: tf.Tensor4D,
    groundTruth: GroundTruth[],
    predictedBoxes: GroundTruthWithGridPosition[],
    reshapedImgDims: IDimensions,
    config: TinyYolov2TrainableConfig
  ) {
    this._config = config
    this._reshapedImgDims = new Dimensions(reshapedImgDims.width, reshapedImgDims.height)
    this._outputTensor = outputTensor
    this._predictedBoxes = predictedBoxes

    this.validateGroundTruthBoxes(groundTruth)
    this._groundTruth = this.assignGroundTruthToAnchors(groundTruth)

    const groundTruthMask = this.createGroundTruthMask()
    const { coordBoxOffsetMask, coordBoxSizeMask, scoreMask } = this.createCoordAndScoreMasks()

    this.noObjectLossMask = tf.tidy(() => tf.mul(scoreMask, tf.sub(tf.scalar(1), groundTruthMask))) as tf.Tensor4D
    this.objectLossMask = tf.tidy(() => tf.mul(scoreMask, groundTruthMask)) as tf.Tensor4D
    this.coordBoxOffsetMask = tf.tidy(() => tf.mul(coordBoxOffsetMask, groundTruthMask)) as tf.Tensor4D
    this.coordBoxSizeMask = tf.tidy(() => tf.mul(coordBoxSizeMask, groundTruthMask)) as tf.Tensor4D

    const classScoresMask = tf.tidy(() => tf.sub(tf.scalar(1), coordBoxOffsetMask.add(coordBoxSizeMask).add(scoreMask)))
    this.groundTruthClassScoresMask = tf.tidy(() => tf.mul(classScoresMask, groundTruthMask)) as tf.Tensor4D
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

  public toOutputTensorShape(tensor: tf.Tensor) {
    return tf.tidy(() => tensor.reshape([1, this.numCells, this.numCells, this.gridCellEncodingSize]))
  }

  public computeLoss() {
    return tf.tidy(() => {

      const noObjectLoss = this.computeNoObjectLoss()
      const objectLoss = this.computeObjectLoss()
      const coordLoss = this.computeCoordLoss()
      const classLoss = this.withClassScores
        ? this.computeClassLoss()
        : tf.scalar(0)

      const totalLoss: tf.Tensor<tf.Rank.R0> = tf.tidy(() => noObjectLoss.add(objectLoss).add(coordLoss).add(classLoss))

      return {
        noObjectLoss,
        objectLoss,
        coordLoss,
        classLoss,
        totalLoss
      }
    })
  }

  public computeNoObjectLoss(): tf.Tensor<tf.Rank.R0> {
    return tf.tidy(() =>
      this.computeLossTerm(
        this.config.noObjectScale,
        this.toOutputTensorShape(this.noObjectLossMask),
        tf.sigmoid(this.outputTensor)
      )
    )
  }

  public computeObjectLoss(): tf.Tensor<tf.Rank.R0> {
    return tf.tidy(() =>
      this.computeLossTerm(
        this.config.objectScale,
        this.toOutputTensorShape(this.objectLossMask),
        tf.sub(this.toOutputTensorShape(this.computeIous()), tf.sigmoid(this.outputTensor))
      )
    )
  }

  public computeClassLoss(): tf.Tensor<tf.Rank.R0> {
    return tf.tidy(() => {

      const classLossTensor = tf.tidy(() => {

        const predClassScores = tf.mul(
          tf.softmax(this.outputTensor.reshape([this.numCells, this.numCells, this.numBoxes, this.boxEncodingSize]), 3),
          this.groundTruthClassScoresMask
        )

        const gtClassScores = this.createOneHotClassScoreMask()

        return tf.sub(gtClassScores, predClassScores)
      })

      return this.computeLossTerm(
        this.config.classScale,
        tf.scalar(1),
        classLossTensor as tf.Tensor4D
      )
    })
  }

  public computeCoordLoss(): tf.Tensor<tf.Rank.R0> {
    return tf.tidy(() =>
      this.computeLossTerm(
        this.config.coordScale,
        tf.scalar(1),
        tf.add(this.computeCoordBoxOffsetError(), this.computeCoordBoxSizeError())
      )
    )
  }

  public computeCoordBoxOffsetError(): tf.Tensor4D {
    return tf.tidy(() => {

      const mask = this.toOutputTensorShape(this.coordBoxOffsetMask)
      const gtBoxOffsets = tf.mul(mask, this.toOutputTensorShape(this.computeCoordBoxOffsets()))
      const predBoxOffsets = tf.mul(mask, tf.sigmoid(this.outputTensor))

      return tf.sub(gtBoxOffsets, predBoxOffsets)

    })
  }

  public computeCoordBoxSizeError(): tf.Tensor4D {
    return tf.tidy(() => {

      const mask = this.toOutputTensorShape(this.coordBoxSizeMask)
      const gtBoxSizes = tf.mul(mask, this.toOutputTensorShape(this.computeCoordBoxSizes()))
      const predBoxSizes = tf.mul(mask, this.outputTensor)

      return tf.sub(gtBoxSizes, predBoxSizes)

    })
  }

  private computeLossTerm(scale: number, mask: tf.Tensor<tf.Rank>, lossTensor: tf.Tensor4D): tf.Tensor<tf.Rank.R0> {
    return tf.tidy(() => tf.mul(tf.scalar(scale), this.squaredSumOverMask(mask, lossTensor)))
  }

  private squaredSumOverMask(mask: tf.Tensor<tf.Rank>, lossTensor: tf.Tensor4D): tf.Tensor4D {
    return tf.tidy(() => tf.sum(tf.square(tf.mul(mask, lossTensor))))
  }

  private validateGroundTruthBoxes(groundTruth: GroundTruth[]) {
    groundTruth.forEach(({ x, y, width, height, label }) => {
      if (typeof label !== 'number' || label < 0 || label > (this.config.classes.length - 1)) {
        throw new Error(`invalid ground truth data, expected label to be a number in [0, ${this.config.classes.length - 1}]`)
      }

      if (x < 0 || x > 1 || y < 0 || y > 1 || width < 0 || (x + width) > 1 || height < 0 || (y + height) > 1) {
        throw new Error(`invalid ground truth data, box is out of image boundaries ${JSON.stringify({ x, y, width, height})}`)
      }
    })
  }

  private assignGroundTruthToAnchors(groundTruth: GroundTruth[]) {
    const groundTruthBoxes = groundTruth
      .map(({ x, y, width, height, label }) => ({
        box: new Rect(x, y, width, height),
        label
      }))

    return groundTruthBoxes.map(({ box, label }) => {
      const { left, top, bottom, right, x, y, width, height } = box.rescale(this.reshapedImgDims)

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

      return { row, col, anchor, box, label }
    })
  }

  private createGroundTruthMask() {

    const mask = tf.zeros([this.numCells, this.numCells, this.numBoxes, this.boxEncodingSize])
    const buf = mask.buffer()

    this.groundTruth.forEach(({ row, col, anchor }) => {
      for (let i = 0; i < this.boxEncodingSize; i++) {
        buf.set(1, row, col, anchor, i)
      }
    })

    return mask
  }

  private createCoordAndScoreMasks() {
    return tf.tidy(() => {

      const coordBoxOffsetMask = tf.zeros([this.numCells, this.numCells, this.numBoxes, this.boxEncodingSize])
      const coordBoxSizeMask = tf.zeros([this.numCells, this.numCells, this.numBoxes, this.boxEncodingSize])
      const scoreMask = tf.zeros([this.numCells, this.numCells, this.numBoxes, this.boxEncodingSize])

      const coordBoxOffsetBuf = coordBoxOffsetMask.buffer()
      const coordBoxSizeBuf = coordBoxSizeMask.buffer()
      const scoreBuf = scoreMask.buffer()

      for (let row = 0; row < this.numCells; row++) {
        for (let col = 0; col < this.numCells; col++) {
          for (let anchor = 0; anchor < this.numBoxes; anchor++) {
            coordBoxOffsetBuf.set(1, row, col, anchor, 0)
            coordBoxOffsetBuf.set(1, row, col, anchor, 1)
            coordBoxSizeBuf.set(1, row, col, anchor, 2)
            coordBoxSizeBuf.set(1, row, col, anchor, 3)
            scoreBuf.set(1, row, col, anchor, 4)
          }
        }
      }

      return { coordBoxOffsetMask, coordBoxSizeMask, scoreMask }
    })

  }

  private createOneHotClassScoreMask() {
    const mask = tf.zeros([this.numCells, this.numCells, this.numBoxes, this.boxEncodingSize])
    const buf = mask.buffer()

    const classValuesOffset = 5
    this.groundTruth.forEach(({ row, col, anchor, label }) => {
      buf.set(1, row, col, anchor, classValuesOffset + label)
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

      const anchorOffset = this.boxEncodingSize * anchor
      const scoreValueOffset = 4
      buf.set(boxIou, row, col, anchorOffset + scoreValueOffset)
    })

    return ious
  }

  public computeCoordBoxOffsets() {

    const offsets = tf.zeros([this.numCells, this.numCells, this.numBoxes, this.boxEncodingSize])
    const buf = offsets.buffer()

    this.groundTruth.forEach(({ row, col, anchor, box }) => {
      const { left, top, right, bottom } = box.rescale(this.reshapedImgDims)

      const centerX = (left + right) / 2
      const centerY = (top + bottom) / 2

      const dCenterX = centerX - (col * CELL_SIZE)
      const dCenterY = centerY - (row * CELL_SIZE)

      // inverseSigmoid(0) === -Infinity, inverseSigmoid(1) === Infinity
      //const dx = inverseSigmoid(Math.min(0.999, Math.max(0.001, dCenterX / CELL_SIZE)))
      //const dy = inverseSigmoid(Math.min(0.999, Math.max(0.001, dCenterY / CELL_SIZE)))
      const dx = dCenterX / CELL_SIZE
      const dy = dCenterY / CELL_SIZE

      buf.set(dx, row, col, anchor, 0)
      buf.set(dy, row, col, anchor, 1)
    })

    return offsets
  }

  public computeCoordBoxSizes() {

    const sizes = tf.zeros([this.numCells, this.numCells, this.numBoxes, this.boxEncodingSize])
    const buf = sizes.buffer()

    this.groundTruth.forEach(({ row, col, anchor, box }) => {
      const { width, height } = box.rescale(this.reshapedImgDims)
      const dw = Math.log(width / (this.anchors[anchor].x * CELL_SIZE))
      const dh = Math.log(height / (this.anchors[anchor].y * CELL_SIZE))

      buf.set(dw, row, col, anchor, 2)
      buf.set(dh, row, col, anchor, 3)
    })

    return sizes
  }

}
