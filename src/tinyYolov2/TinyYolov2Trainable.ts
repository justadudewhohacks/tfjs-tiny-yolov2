import * as tf from '@tensorflow/tfjs-core';
import {
  computeReshapedDimensions,
  Dimensions,
  getMediaDimensions,
  imageToSquare,
  Rect,
  toNetInput,
} from 'tfjs-image-recognition-base';

import { TinyYolov2TrainableConfig, validateTrainConfig } from './config';
import { getDefaultBackwardOptions } from './getDefaultBackwardOptions';
import { TinyYolov2 } from './TinyYolov2';
import { TinyYolov2LossFunction } from './TinyYolov2LossFunction';
import { GroundTruth, TinyYolov2BackwardOptions } from './types';

export class TinyYolov2Trainable extends TinyYolov2 {

  private _trainableConfig: TinyYolov2TrainableConfig
  private _optimizer: tf.Optimizer

  constructor(trainableConfig: TinyYolov2TrainableConfig, optimizer: tf.Optimizer) {
    super(trainableConfig)
    this._trainableConfig = validateTrainConfig(trainableConfig)
    this._optimizer = optimizer
  }

  public get trainableConfig(): TinyYolov2TrainableConfig {
    return this._trainableConfig
  }

  public get optimizer(): tf.Optimizer {
    return this._optimizer
  }

  public async backward(
    img: HTMLImageElement | HTMLCanvasElement,
    groundTruth: GroundTruth[],
    inputSize: number,
    options: TinyYolov2BackwardOptions = {}
  ): Promise<tf.Tensor<tf.Rank.R0> | null> {

    const { minBoxSize, reportLosses } = getDefaultBackwardOptions(options)
    const reshapedImgDims = computeReshapedDimensions(getMediaDimensions(img), inputSize)
    const filteredGroundTruthBoxes = this.filterGroundTruthBoxes(groundTruth, reshapedImgDims, minBoxSize)

    if (!filteredGroundTruthBoxes.length) {
      return null
    }

    // square input images before creating tensor to prevent gpu memory overflow bug
    const netInput = await toNetInput(imageToSquare(img, inputSize))

    const loss = this.optimizer.minimize(() => {

      const {
        noObjectLoss,
        objectLoss,
        coordLoss,
        classLoss,
        totalLoss
      } = this.computeLoss(
        this.forwardInput(netInput, inputSize),
        filteredGroundTruthBoxes,
        reshapedImgDims
      )

      if (reportLosses) {
        const losses = {
          totalLoss: totalLoss.dataSync()[0],
          noObjectLoss: noObjectLoss.dataSync()[0],
          objectLoss: objectLoss.dataSync()[0],
          coordLoss: coordLoss.dataSync()[0],
          classLoss: classLoss.dataSync()[0]
        }

        const report = {
          losses,
          numBoxes: filteredGroundTruthBoxes.length,
          inputSize
        }

        reportLosses(report)
      }

      return totalLoss
    }, true)

    return loss
  }

  public computeLoss(outputTensor: tf.Tensor4D, groundTruth: GroundTruth[], reshapedImgDims: Dimensions) {

    const config = validateTrainConfig(this.config)

    const inputSize = Math.max(reshapedImgDims.width, reshapedImgDims.height)

    if (!inputSize) {
      throw new Error(`computeLoss - invalid inputSize: ${inputSize}`)
    }

    const predictedBoxes = this.extractBoxes(outputTensor, reshapedImgDims)

    return tf.tidy(() => {
      const lossFunction = new TinyYolov2LossFunction(outputTensor, groundTruth, predictedBoxes, reshapedImgDims, config)
      return lossFunction.computeLoss()
    })
  }

  public filterGroundTruthBoxes(groundTruth: GroundTruth[], imgDims: Dimensions, minBoxSize: number) {

    const { height: imgHeight, width: imgWidth } = imgDims

    return groundTruth.filter(({ x, y, width, height }) => {
      const box = (new Rect(x, y, width, height))
        .rescale({ height: imgHeight, width: imgWidth })

      const isTooTiny = box.width < minBoxSize || box.height < minBoxSize
      return !isTooTiny
    })
  }

  public async load(weightsOrUrl: Float32Array | string | undefined): Promise<void> {
    await super.load(weightsOrUrl)
    this.variable()
  }
}