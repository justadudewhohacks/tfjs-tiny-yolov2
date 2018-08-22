import * as tf from '@tensorflow/tfjs-core';

import { Rect } from '../src/Rect';
import { createFakeLossFunction, createFakeLossFunctionMd, expectTensorToBeZeros } from './utils';

describe('TinyYolov2LossFunction', () => {

  describe('no ground truth', () => {

    it('ground truth masks empty', () => tf.tidy(() => {

      const lossFunction = createFakeLossFunctionMd([], [])

      expectTensorToBeZeros(lossFunction.objectLossMask)
      expectTensorToBeZeros(lossFunction.coordLossMask)
      expectTensorToBeZeros(lossFunction.groundTruthClassScoresMask)
    }))

  })

  describe('masks initialized correctly', () => {

    const groundTruth = { ...new Rect(0, 0, 0.25, 0.5), classLabel: 0 }

    it('noObjectLossMask', () => tf.tidy(() => {

      const lossFunction = createFakeLossFunction(2, [groundTruth], [])
      const { noObjectLossMask } = lossFunction
      expect(noObjectLossMask.shape).toEqual([2, 2, 7])

      for (let y = 0; y < 2; y++) {
        for (let x = 0; x < 2; x++) {
          for (let i = 0; i < 7; i++) {
            expect(noObjectLossMask.get(y, x, i)).toEqual((i === 4 && (y !== 0 || x !== 0)) ? 1 : 0)
          }
        }
      }

    }))

    it('objectLossMask', () => tf.tidy(() => {

      const lossFunction = createFakeLossFunction(2, [groundTruth], [])
      const { objectLossMask } = lossFunction
      expect(objectLossMask.shape).toEqual([2, 2, 7])

      for (let y = 0; y < 2; y++) {
        for (let x = 0; x < 2; x++) {
          for (let i = 0; i < 7; i++) {
            expect(objectLossMask.get(y, x, i)).toEqual((i === 4 && y === 0 && x === 0) ? 1 : 0)

          }
        }
      }
    }))

    it('coordLossMask', () => tf.tidy(() => {

      const lossFunction = createFakeLossFunction(2, [groundTruth], [])
      const { coordLossMask } = lossFunction
      expect(coordLossMask.shape).toEqual([2, 2, 7])

      for (let y = 0; y < 2; y++) {
        for (let x = 0; x < 2; x++) {
          for (let i = 0; i < 7; i++) {
            expect(coordLossMask.get(y, x, i)).toEqual((i < 4 && y === 0 && x === 0) ? 1 : 0)
          }
        }
      }
    }))

    it('groundTruthClassScoresMask', () => tf.tidy(() => {

      const lossFunction = createFakeLossFunction(2, [groundTruth], [])
      const { groundTruthClassScoresMask } = lossFunction
      expect(groundTruthClassScoresMask.shape).toEqual([2, 2, 7])

      for (let y = 0; y < 2; y++) {
        for (let x = 0; x < 2; x++) {
          for (let i = 0; i < 7; i++) {
            expect(groundTruthClassScoresMask.get(y, x, i)).toEqual(((i === 5 || i === 6) && y === 0 && x === 0) ? 1 : 0)
          }
        }
      }
    }))

  })

  describe('computeObjectLoss', () => {

    const groundTruth = { ...new Rect(0, 0, 0.25, 0.5), classLabel: 0 }
    const predictedBox = { box: new Rect(0, 0, 0.25, 0.25).toBoundingBox(), classLabel: 0, row: 0, col: 0, anchor: 0 }

    it('single box iou of 0.5', () => tf.tidy(() => {

      const lossFunction = createFakeLossFunction(2, [groundTruth], [predictedBox])
      const buf = lossFunction.outputTensor.buffer()
      buf.set(0, 0, predictedBox.row, predictedBox.col, 4)

      const objectLoss = lossFunction.computeObjectLoss()
      expectTensorToBeZeros(objectLoss)

    }))

    it('two box ious of 0.5', () => tf.tidy(() => {

      const groundTruth2 = { ...new Rect(0.5, 0.5, 0.25, 0.5), classLabel: 0 }
      const predictedBox2 = { box: new Rect(0.5, 0.5, 0.25, 0.25).toBoundingBox(), classLabel: 0, row: 1, col: 1, anchor: 0 }

      const lossFunction = createFakeLossFunction(2, [groundTruth, groundTruth2], [predictedBox, predictedBox2])
      const buf = lossFunction.outputTensor.buffer()

      buf.set(0, 0, predictedBox.row, predictedBox.col, 4)
      buf.set(0, 0, predictedBox2.row, predictedBox2.col, 4)

      const objectLoss = lossFunction.computeObjectLoss()
      expectTensorToBeZeros(objectLoss)

    }))

  })


})