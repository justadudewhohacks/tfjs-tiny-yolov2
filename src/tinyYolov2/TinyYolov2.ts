import * as tf from '@tensorflow/tfjs-core';

import { BoundingBox } from '../BoundingBox';
import { convLayer } from '../commons/convLayer';
import { NeuralNetwork } from '../commons/NeuralNetwork';
import { nonMaxSuppression } from '../commons/nonMaxSuppression';
import { normalize } from '../commons/normalize';
import { NetInput } from '../NetInput';
import { ObjectDetection } from '../ObjectDetection';
import { toNetInput } from '../toNetInput';
import { Dimensions, TNetInput } from '../types';
import { sigmoid } from '../utils';
import { TinyYolov2Config, validateConfig } from './config';
import { INPUT_SIZES } from './const';
import { convWithBatchNorm } from './convWithBatchNorm';
import { extractParams } from './extractParams';
import { getDefaultParams } from './getDefaultParams';
import { loadQuantizedParams } from './loadQuantizedParams';
import { NetParams, TinyYolov2ForwardParams } from './types';

export class TinyYolov2 extends NeuralNetwork<NetParams> {

  private _config: TinyYolov2Config

  constructor(config: TinyYolov2Config) {
    super('TinyYolov2')
    validateConfig(config)
    this._config = config
  }

  public get config(): TinyYolov2Config {
    return this._config
  }

  public get withClassScores(): boolean {
    return this.config.withClassScores || this.config.classes.length > 1
  }

  public get boxEncodingSize(): number{
    return 5 + (this.withClassScores ? this.config.classes.length : 0)
  }

  public forwardInput(input: NetInput, inputSize: number): tf.Tensor4D {

    const { params } = this

    if (!params) {
      throw new Error('TinyYolov2 - load model before inference')
    }

    const out = tf.tidy(() => {

      let batchTensor = input.toBatchTensor(inputSize, false)
      batchTensor = this.config.meanRgb
        ? normalize(batchTensor, this.config.meanRgb)
        : batchTensor
      batchTensor = batchTensor.div(tf.scalar(256)) as tf.Tensor4D

      let out = convWithBatchNorm(batchTensor, params.conv0)
      out = tf.maxPool(out, [2, 2], [2, 2], 'same')
      out = convWithBatchNorm(out, params.conv1)
      out = tf.maxPool(out, [2, 2], [2, 2], 'same')
      out = convWithBatchNorm(out, params.conv2)
      out = tf.maxPool(out, [2, 2], [2, 2], 'same')
      out = convWithBatchNorm(out, params.conv3)
      out = tf.maxPool(out, [2, 2], [2, 2], 'same')
      out = convWithBatchNorm(out, params.conv4)
      out = tf.maxPool(out, [2, 2], [2, 2], 'same')
      out = convWithBatchNorm(out, params.conv5)
      out = tf.maxPool(out, [2, 2], [1, 1], 'same')
      out = convWithBatchNorm(out, params.conv6)
      out = convWithBatchNorm(out, params.conv7)
      out = convLayer(out, params.conv8, 'valid', false)

      return out
    })

    return out
  }

  public async forward(input: TNetInput, inputSize: number): Promise<tf.Tensor4D> {
    return await this.forwardInput(await toNetInput(input, true, true), inputSize)
  }

  public async detect(input: TNetInput, forwardParams: TinyYolov2ForwardParams = {}): Promise<ObjectDetection[]> {

    const { inputSize: _inputSize, scoreThreshold } = getDefaultParams(forwardParams)

    const inputSize = typeof _inputSize === 'string'
      ? INPUT_SIZES[_inputSize]
      : _inputSize

    if (typeof inputSize !== 'number') {
      throw new Error(`TinyYolov2 - unknown inputSize: ${inputSize}, expected number or one of xs | sm | md | lg`)
    }

    const netInput = await toNetInput(input, true)
    const out = await this.forwardInput(netInput, inputSize)
    const out0 = tf.tidy(() => tf.unstack(out)[0].expandDims()) as tf.Tensor4D

    const inputDimensions = {
      width: netInput.getInputWidth(0),
      height: netInput.getInputHeight(0)
    }

    const results = this.extractBoxes(out0, scoreThreshold, netInput.getReshapedInputDimensions(0))
    out.dispose()
    out0.dispose()

    const boxes = results.map(res => res.box)
    const scores = results.map(res => res.score)
    const classNames = results.map(res => res.className)

    const indices = nonMaxSuppression(
      boxes.map(box => box.rescale(inputSize)),
      scores,
      this.config.iouThreshold,
      true
    )

    const detections = indices.map(idx =>
      new ObjectDetection(
        scores[idx],
        classNames[idx],
        boxes[idx].toRect(),
        inputDimensions
      )
    )

    return detections
  }

  public extractBoxes(outputTensor: tf.Tensor4D, scoreThreshold: number, inputBlobDimensions: Dimensions) {

    const { width, height } = inputBlobDimensions
    const inputSize = Math.max(width, height)
    const correctionFactorX = inputSize / width
    const correctionFactorY = inputSize / height

    const numCells = outputTensor.shape[1]
    const numBoxes = this.config.anchors.length

    const [boxesTensor, scoresTensor, classesTensor] = tf.tidy(() => {
      const reshaped = outputTensor.reshape([numCells, numCells, numBoxes, this.boxEncodingSize])

      const boxes = reshaped.slice([0, 0, 0, 0], [numCells, numCells, numBoxes, 4])
      const scores = reshaped.slice([0, 0, 0, 4], [numCells, numCells, numBoxes, 1])
      const classes = this.withClassScores
        ? reshaped.slice([0, 0, 0, 5], [numCells, numCells, numBoxes, this.config.classes.length])
        : tf.scalar(0)
      return [boxes, scores, classes]
    })

    const results = []

    for (let row = 0; row < numCells; row ++) {
      for (let col = 0; col < numCells; col ++) {
        for (let anchor = 0; anchor < numBoxes; anchor ++) {
          const score = sigmoid(scoresTensor.get(row, col, anchor, 0))
          if (!scoreThreshold || score > scoreThreshold) {
            const ctX = ((col + sigmoid(boxesTensor.get(row, col, anchor, 0))) / numCells) * correctionFactorX
            const ctY = ((row + sigmoid(boxesTensor.get(row, col, anchor, 1))) / numCells) * correctionFactorY
            const width = ((Math.exp(boxesTensor.get(row, col, anchor, 2)) * this.config.anchors[anchor].x) / numCells) * correctionFactorX
            const height = ((Math.exp(boxesTensor.get(row, col, anchor, 3)) * this.config.anchors[anchor].y) / numCells) * correctionFactorY

            const x = (ctX - (width / 2))
            const y = (ctY - (height / 2))

            const pos = { row, col, anchor }
            const classScores = this.withClassScores
              ? this.extractClassScores(classesTensor as tf.Tensor4D, score, pos)
              : [score]

            const { classScore, className } = classScores
              .map((classScore, idx) => ({
                className: this.config.classes[idx],
                classScore
              }))
              .reduce((max, curr) => max.classScore > curr.classScore ? max : curr)

            results.push({
              box: new BoundingBox(x, y, x + width, y + height),
              score: classScore,
              className,
              ...pos
            })
          }
        }
      }
    }

    boxesTensor.dispose()
    scoresTensor.dispose()

    return results
  }

  public extractClassScores(classesTensor: tf.Tensor4D, score: number, pos: { row: number, col: number, anchor: number }) {
    const { row, col, anchor } = pos
    const classesData = Array(this.config.classes.length).fill(0).map((_, i) => classesTensor.get(row, col, anchor, i))

    const maxClass = classesData.reduce((max, c) => max > c ? max : c)
    const classes = classesData.map(c => Math.exp(c - maxClass))
    const sum = classes.reduce((sum, c) => sum + c)

    return classes.map(c => (c * score) / sum)
  }

  protected loadQuantizedParams(modelUri: string | undefined) {
    if (!modelUri) {
      throw new Error('loadQuantizedParams - please specify the modelUri')
    }

    return loadQuantizedParams(modelUri, this.config.withSeparableConvs)
  }

  protected extractParams(weights: Float32Array) {
    return extractParams(weights, this.config.withSeparableConvs, this.boxEncodingSize)
  }
}