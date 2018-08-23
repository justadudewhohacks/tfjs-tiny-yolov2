import * as tf from '@tensorflow/tfjs-core';

import { Rect } from '../src/Rect';
import { GridPosition, GroundTruth } from '../src/tinyYolov2/types';
import { inverseSigmoid } from '../src/utils';
import { createFakeLossFunction, createFakeLossFunctionMd, expectTensorToBeZeros } from './utils';

describe('TinyYolov2LossFunction', () => {

  describe('no ground truth', () => {

    it('ground truth masks empty', () => tf.tidy(() => {

      const lossFunction = createFakeLossFunctionMd([], [])

      expectTensorToBeZeros(lossFunction.objectLossMask)
      expectTensorToBeZeros(lossFunction.coordBoxOffsetMask)
      expectTensorToBeZeros(lossFunction.coordBoxSizeMask)
      expectTensorToBeZeros(lossFunction.groundTruthClassScoresMask)
    }))

  })

  describe('masks initialized correctly', () => {

    describe('single anchor', () => {

      const groundTruth = { ...new Rect(0, 0, 0.25, 0.5), classLabel: 0 }

      it('noObjectLossMask', () => tf.tidy(() => {

        const lossFunction = createFakeLossFunction(2, [groundTruth], [])
        const { noObjectLossMask } = lossFunction
        expect(noObjectLossMask.shape).toEqual([2, 2, 1, 7])

        for (let y = 0; y < 2; y++) {
          for (let x = 0; x < 2; x++) {
            for (let a = 0; a < 1; a++) {
              for (let i = 0; i < 7; i++) {
                const isGt = y === 0 && x === 0 && a === 0
                expect(noObjectLossMask.get(y, x, a, i)).toEqual((i === 4 && !isGt) ? 1 : 0)
              }
            }
          }
        }

      }))

      it('objectLossMask', () => tf.tidy(() => {

        const lossFunction = createFakeLossFunction(2, [groundTruth], [])
        const { objectLossMask } = lossFunction
        expect(objectLossMask.shape).toEqual([2, 2, 1, 7])

        for (let y = 0; y < 2; y++) {
          for (let x = 0; x < 2; x++) {
            for (let a = 0; a < 1; a++) {
              for (let i = 0; i < 7; i++) {
                const isGt = y === 0 && x === 0 && a === 0
                expect(objectLossMask.get(y, x, a, i)).toEqual((i === 4 && isGt) ? 1 : 0)
              }
            }
          }
        }
      }))

      it('coordBoxOffsetMask', () => tf.tidy(() => {

        const lossFunction = createFakeLossFunction(2, [groundTruth], [])
        const { coordBoxOffsetMask } = lossFunction
        expect(coordBoxOffsetMask.shape).toEqual([2, 2, 1, 7])

        for (let y = 0; y < 2; y++) {
          for (let x = 0; x < 2; x++) {
            for (let a = 0; a < 1; a++) {
              for (let i = 0; i < 7; i++) {
                const isGt = y === 0 && x === 0 && a === 0
                const isCoordOffset = i === 0 || i === 1
                expect(coordBoxOffsetMask.get(y, x, a, i)).toEqual((isCoordOffset && isGt) ? 1 : 0)
              }
            }
          }
        }
      }))

      it('coordBoxOffsetMask', () => tf.tidy(() => {

        const lossFunction = createFakeLossFunction(2, [groundTruth], [])
        const { coordBoxSizeMask } = lossFunction
        expect(coordBoxSizeMask.shape).toEqual([2, 2, 1, 7])

        for (let y = 0; y < 2; y++) {
          for (let x = 0; x < 2; x++) {
            for (let a = 0; a < 1; a++) {
              for (let i = 0; i < 7; i++) {
                const isGt = y === 0 && x === 0 && a === 0
                const isCoordOffset = i === 2 || i === 3
                expect(coordBoxSizeMask.get(y, x, a, i)).toEqual((isCoordOffset && isGt) ? 1 : 0)
              }
            }
          }
        }

      }))

      it('groundTruthClassScoresMask', () => tf.tidy(() => {

        const lossFunction = createFakeLossFunction(2, [groundTruth], [])
        const { groundTruthClassScoresMask } = lossFunction
        expect(groundTruthClassScoresMask.shape).toEqual([2, 2, 1, 7])

        for (let y = 0; y < 2; y++) {
          for (let x = 0; x < 2; x++) {
            for (let a = 0; a < 1; a++) {
              for (let i = 0; i < 7; i++) {
                const isGt = y === 0 && x === 0 && a === 0
                expect(groundTruthClassScoresMask.get(y, x, a, i)).toEqual(((i === 5 || i === 6) && isGt) ? 1 : 0)
              }
            }
          }
        }

      }))

    })

    describe('multiple anchors', () => {

      const groundTruth = { ...new Rect(0, 0, 0.25, 0.5), classLabel: 0 }
      const anchors = [{ x: 4, y: 4 }, { x: 1, y: 1 }, { x: 2, y: 2 }]

      it('noObjectLossMask', () => tf.tidy(() => {

        const lossFunction = createFakeLossFunction(2, [groundTruth], [], { anchors })
        const { noObjectLossMask } = lossFunction
        expect(noObjectLossMask.shape).toEqual([2, 2, 3, 7])

        for (let y = 0; y < 2; y++) {
          for (let x = 0; x < 2; x++) {
            for (let a = 0; a < 3; a++) {
              for (let i = 0; i < 7; i++) {
                const isGt = y === 0 && x === 0 && a === 1
                expect(noObjectLossMask.get(y, x, a, i)).toEqual((i === 4 && !isGt) ? 1 : 0)
              }
            }
          }
        }

      }))

    })


  })

  describe('assignGroundTruthToAnchors', () => {

    describe('square image', () => {

      it('assigns boxes correctly for single anchor', () => tf.tidy(() => {

        const numCells = 10
        const boxSize = 1 / numCells
        const anchors = [{ x: 1, y: 1 }]
        const groundTruthBoxes = [
          { ...new Rect(0, 0, boxSize, boxSize), classLabel: 0 },
          { ...new Rect(0.9, 0.9, boxSize, boxSize), classLabel: 0 },
          { ...new Rect(0.1, 0.9, boxSize, boxSize), classLabel: 0 },
          { ...new Rect(0.9, 0.1, boxSize, boxSize), classLabel: 0 }
        ]

        const { groundTruth } = createFakeLossFunction(numCells, groundTruthBoxes, [], { anchors })

        expect(groundTruth.length).toEqual(4)
        expect(groundTruth[0].row).toEqual(0)
        expect(groundTruth[0].col).toEqual(0)
        expect(groundTruth[0].anchor).toEqual(0)
        expect(groundTruth[1].row).toEqual(9)
        expect(groundTruth[1].col).toEqual(9)
        expect(groundTruth[1].anchor).toEqual(0)
        expect(groundTruth[2].row).toEqual(9)
        expect(groundTruth[2].col).toEqual(1)
        expect(groundTruth[2].anchor).toEqual(0)
        expect(groundTruth[3].row).toEqual(1)
        expect(groundTruth[3].col).toEqual(9)
        expect(groundTruth[3].anchor).toEqual(0)

      }))

      it('assigns anchors correctly', () => tf.tidy(() => {

        const numCells = 10
        const anchors = [{ x: 1, y: 1 }, { x: 2, y: 2 }, { x: 4, y: 4 }, { x: 2, y: 4 }]
        const groundTruthBoxes = [
          { ...new Rect(0, 0, 0.1, 0.1), classLabel: 0 },
          { ...new Rect(0, 0, 0.2, 0.2), classLabel: 0 },
          { ...new Rect(0, 0, 0.4, 0.4), classLabel: 0 },
          { ...new Rect(0, 0, 0.2, 0.4), classLabel: 0 }
        ]

        const { groundTruth } = createFakeLossFunction(numCells, groundTruthBoxes, [], { anchors })

        expect(groundTruth.length).toEqual(4)
        expect(groundTruth[0].anchor).toEqual(0)
        expect(groundTruth[1].anchor).toEqual(1)
        expect(groundTruth[2].anchor).toEqual(2)
        expect(groundTruth[3].anchor).toEqual(3)

      }))

    })

  })

  describe('computeCoordBoxOffsets', () => {

    it('computes correct offsets for single anchor', () => tf.tidy(() => {

      const anchors = [{ x: 1, y: 1 }]
      const numCells = 2

      function testComputeBoxAdjusments(groundTruth: GroundTruth, { row, col, anchor }: GridPosition) {
        const offsets = createFakeLossFunction(numCells, [groundTruth], [], { anchors }).computeCoordBoxOffsets()
        expect((col + offsets.get(row, col, anchor, 0)) / numCells).toBeCloseTo(groundTruth.x + (groundTruth.width / 2), 2, 'x')
        expect((row + offsets.get(row, col, anchor, 1)) / numCells).toBeCloseTo(groundTruth.y + (groundTruth.height / 2), 2, 'y')
      }

      testComputeBoxAdjusments({ ...new Rect(0, 0, 0.25, 0.25), classLabel: 0 } as GroundTruth, { row: 0, col: 0, anchor: 0 })
      testComputeBoxAdjusments({ ...new Rect(0, 0, 0.5, 0.5), classLabel: 0 } as GroundTruth, { row: 0, col: 0, anchor: 0 })
      testComputeBoxAdjusments({ ...new Rect(0.25, 0.25, 0.25, 0.25), classLabel: 0 } as GroundTruth, { row: 0, col: 0, anchor: 0 })
      testComputeBoxAdjusments({ ...new Rect(0.5, 0.5, 0.25, 0.25), classLabel: 0 } as GroundTruth, { row: 1, col: 1, anchor: 0 })
      testComputeBoxAdjusments({ ...new Rect(0.75, 0.75, 0.25, 0.25), classLabel: 0 } as GroundTruth, { row: 1, col: 1, anchor: 0 })
      testComputeBoxAdjusments({ ...new Rect(0.25, 0.75, 0.25, 0.25), classLabel: 0 } as GroundTruth, { row: 1, col: 0, anchor: 0 })
      testComputeBoxAdjusments({ ...new Rect(0.75, 0.25, 0.25, 0.25), classLabel: 0 } as GroundTruth, { row: 0, col: 1, anchor: 0 })
    }))

  })

  describe('computeCoordBoxSizes', () => {

    it('computes correct box sizes single anchor', () => tf.tidy(() => {

      const anchors = [{ x: 1, y: 1 }]
      const numCells = 2

      function testComputeBoxAdjusments(groundTruth: GroundTruth, { row, col, anchor }: GridPosition) {
        const boxAdjustments = createFakeLossFunction(numCells, [groundTruth], [], { anchors }).computeCoordBoxSizes()
        expect((Math.exp(boxAdjustments.get(row, col, anchor, 2)) * anchors[anchor].x) / numCells).toBeCloseTo(groundTruth.width, 2, 'width')
        expect((Math.exp(boxAdjustments.get(row, col, anchor, 3)) * anchors[anchor].y) / numCells).toBeCloseTo(groundTruth.height, 2, 'height')
      }

      testComputeBoxAdjusments({ ...new Rect(0, 0, 0.25, 0.25), classLabel: 0 } as GroundTruth, { row: 0, col: 0, anchor: 0 })
      testComputeBoxAdjusments({ ...new Rect(0, 0, 0.5, 0.5), classLabel: 0 } as GroundTruth, { row: 0, col: 0, anchor: 0 })
      testComputeBoxAdjusments({ ...new Rect(0.25, 0.25, 0.25, 0.25), classLabel: 0 } as GroundTruth, { row: 0, col: 0, anchor: 0 })
      testComputeBoxAdjusments({ ...new Rect(0.5, 0.5, 0.25, 0.25), classLabel: 0 } as GroundTruth, { row: 1, col: 1, anchor: 0 })
      testComputeBoxAdjusments({ ...new Rect(0.75, 0.75, 0.25, 0.25), classLabel: 0 } as GroundTruth, { row: 1, col: 1, anchor: 0 })
      testComputeBoxAdjusments({ ...new Rect(0.25, 0.75, 0.25, 0.25), classLabel: 0 } as GroundTruth, { row: 1, col: 0, anchor: 0 })
      testComputeBoxAdjusments({ ...new Rect(0.75, 0.25, 0.25, 0.25), classLabel: 0 } as GroundTruth, { row: 0, col: 1, anchor: 0 })
    }))

  })

  describe('computeObjectLoss', () => {

    it('single box iou of 0.5', () => tf.tidy(() => {

      const groundTruth = { ...new Rect(0, 0, 0.25, 0.5), classLabel: 0 }
      const predictedBox = { box: new Rect(0, 0, 0.25, 0.25).toBoundingBox(), classLabel: 0, row: 0, col: 0, anchor: 0 }

      const lossFunction = createFakeLossFunction(2, [groundTruth], [predictedBox])
      const buf = lossFunction.outputTensor.buffer()
      buf.set(0, 0, predictedBox.row, predictedBox.col, 4)

      expect(lossFunction.computeObjectLoss().dataSync()[0]).toEqual(0)

    }))

    it('two box ious of 0.5', () => tf.tidy(() => {

      const groundTruth = { ...new Rect(0, 0, 0.25, 0.5), classLabel: 0 }
      const predictedBox = { box: new Rect(0, 0, 0.25, 0.25).toBoundingBox(), classLabel: 0, row: 0, col: 0, anchor: 0 }
      const groundTruth2 = { ...new Rect(0.5, 0.5, 0.25, 0.5), classLabel: 0 }
      const predictedBox2 = { box: new Rect(0.5, 0.5, 0.25, 0.25).toBoundingBox(), classLabel: 0, row: 1, col: 1, anchor: 0 }

      const lossFunction = createFakeLossFunction(2, [groundTruth, groundTruth2], [predictedBox, predictedBox2])
      const buf = lossFunction.outputTensor.buffer()

      buf.set(0, 0, predictedBox.row, predictedBox.col, 4)
      buf.set(0, 0, predictedBox2.row, predictedBox2.col, 4)

      expect(lossFunction.computeObjectLoss().dataSync()[0]).toEqual(0)

    }))

    it('single box no overlap', () => tf.tidy(() => {

      const groundTruth = { ...new Rect(0, 0, 0.25, 0.25), classLabel: 0 }
      const predictedBox = { box: new Rect(0.5, 0.5, 0.25, 0.25).toBoundingBox(), classLabel: 0, row: 0, col: 0, anchor: 0 }

      const lossFunction = createFakeLossFunction(2, [groundTruth], [predictedBox])
      const buf = lossFunction.outputTensor.buffer()
      buf.set(inverseSigmoid(1), 0, predictedBox.row, predictedBox.col, 4)

      expect(lossFunction.computeObjectLoss().dataSync()[0]).toEqual(5)

    }))

    it('single box no overlap at anchor 1', () => tf.tidy(() => {

      const anchors = [{ x: 2, y: 2 }, { x: 1, y: 1 }, { x: 4, y: 4 }, { x: 2, y: 4 }]
      const groundTruth = { ...new Rect(0, 0, 0.25, 0.25), classLabel: 0 }
      const predictedBox = { box: new Rect(0.5, 0.5, 0.25, 0.25).toBoundingBox(), classLabel: 0, row: 0, col: 0, anchor: 1 }

      const lossFunction = createFakeLossFunction(2, [groundTruth], [predictedBox], { anchors })
      const buf = lossFunction.outputTensor.buffer()

      const scoreOffset = lossFunction.boxEncodingSize + 4
      buf.set(inverseSigmoid(1), 0, predictedBox.row, predictedBox.col, scoreOffset)

      expect(lossFunction.computeObjectLoss().dataSync()[0]).toEqual(5)

    }))

    it('single box no overlap at grid (1, 1)', () => tf.tidy(() => {

      const groundTruth = { ...new Rect(0.5, 0.5, 0.25, 0.25), classLabel: 0 }
      const predictedBox = { box: new Rect(0.75, 0.75, 0.25, 0.25).toBoundingBox(), classLabel: 0, row: 1, col: 1, anchor: 0 }

      const lossFunction = createFakeLossFunction(2, [groundTruth], [predictedBox])
      const buf = lossFunction.outputTensor.buffer()
      buf.set(inverseSigmoid(1), 0, predictedBox.row, predictedBox.col, 4)

      expect(lossFunction.computeObjectLoss().dataSync()[0]).toEqual(5)

    }))

    it('single box no overlap at grid (9, 9)', () => tf.tidy(() => {

      const groundTruth = { ...new Rect(0.9, 0.9, 0.05, 0.05), classLabel: 0 }
      const predictedBox = { box: new Rect(0.95, 0.95, 0.05, 0.05).toBoundingBox(), classLabel: 0, row: 9, col: 9, anchor: 0 }

      const lossFunction = createFakeLossFunction(10, [groundTruth], [predictedBox])
      const buf = lossFunction.outputTensor.buffer()
      buf.set(inverseSigmoid(1), 0, predictedBox.row, predictedBox.col, 4)

      expect(lossFunction.computeObjectLoss().dataSync()[0]).toEqual(5)

    }))

  })

  describe('computeCoordBoxOffsetError', () => {

    describe('error should be zero', () => {

      it('single box at grid (0, 0, 0)', () => tf.tidy(() => {

        const numCells = 2
        const anchor = { x: 2, y: 4 }
        const groundTruth = { ...new Rect(0, 0.125, 0.5, 0.25), classLabel: 0 }
        const lossFunction = createFakeLossFunction(numCells, [groundTruth], [], { anchors: [anchor]})
        const buf = lossFunction.outputTensor.buffer()

        const dx = inverseSigmoid(0.5)
        const dy = inverseSigmoid(0.5)
        buf.set(dx, 0, 0, 0, 0)
        buf.set(dy, 0, 0, 0, 1)

        expect(tf.sum(lossFunction.computeCoordBoxOffsetError()).dataSync()[0]).toBeCloseTo(0, 2)

      }))

      it('single box at grid (1, 1, 0)', () => tf.tidy(() => {

        const numCells = 2
        const anchor = { x: 2, y: 4 }
        const groundTruth = { ...new Rect(0.5, 0.5, 0.5, 0.25), classLabel: 0 }
        const lossFunction = createFakeLossFunction(numCells, [groundTruth], [], { anchors: [anchor]})
        const buf = lossFunction.outputTensor.buffer()

        const dx = inverseSigmoid(0.5)
        const dy = inverseSigmoid(0.25)
        buf.set(dx, 0, 1, 1, 0)
        buf.set(dy, 0, 1, 1, 1)

        expect(tf.sum(lossFunction.computeCoordBoxOffsetError()).dataSync()[0]).toBeCloseTo(0, 2)

      }))

      it('single box at grid (1, 1, 1)', () => tf.tidy(() => {

        const numCells = 2
        const anchor = { x: 2, y: 4 }
        const anchors = [{ x: 100, y: 100 }, anchor]
        const groundTruth = { ...new Rect(0.5, 0.5, 0.5, 0.25), classLabel: 0 }
        const lossFunction = createFakeLossFunction(numCells, [groundTruth], [], { anchors })
        const buf = lossFunction.outputTensor.buffer()

        const dx = inverseSigmoid(0.5)
        const dy = inverseSigmoid(0.25)
        buf.set(dx, 0, 1, 1, 0  + lossFunction.boxEncodingSize)
        buf.set(dy, 0, 1, 1, 1  + lossFunction.boxEncodingSize)

        expect(tf.sum(lossFunction.computeCoordBoxOffsetError()).dataSync()[0]).toBeCloseTo(0, 2)

      }))

    })

    describe('should compute error', () => {

      it('single box at grid (0, 0, 0)', () => tf.tidy(() => {

        const numCells = 2
        const anchor = { x: 2, y: 4 }
        const groundTruth = { ...new Rect(0, 0.125, 0.5, 0.25), classLabel: 0 }
        const lossFunction = createFakeLossFunction(numCells, [groundTruth], [], { anchors: [anchor]})
        const buf = lossFunction.outputTensor.buffer()

        const dx = 2 + inverseSigmoid(0.5)
        const dy = 2 + inverseSigmoid(0.5)
        buf.set(dx, 0, 0, 0, 0)
        buf.set(dy, 0, 0, 0, 1)

        expect(Math.abs(tf.sum(lossFunction.computeCoordBoxOffsetError()).dataSync()[0])).toBeGreaterThan(0)

      }))

      it('single box at grid (1, 1, 0)', () => tf.tidy(() => {

        const numCells = 2
        const anchor = { x: 2, y: 4 }
        const groundTruth = { ...new Rect(0.5, 0.5, 0.5, 0.25), classLabel: 0 }
        const lossFunction = createFakeLossFunction(numCells, [groundTruth], [], { anchors: [anchor]})
        const buf = lossFunction.outputTensor.buffer()

        const dx = 2 + inverseSigmoid(0.5)
        const dy = 2 + inverseSigmoid(0.25)
        buf.set(dx, 0, 1, 1, 0)
        buf.set(dy, 0, 1, 1, 1)

        expect(Math.abs(tf.sum(lossFunction.computeCoordBoxOffsetError()).dataSync()[0])).toBeGreaterThan(0)

      }))

      it('single box at grid (1, 1, 1)', () => tf.tidy(() => {

        const numCells = 2
        const anchor = { x: 2, y: 4 }
        const anchors = [{ x: 100, y: 100 }, anchor]
        const groundTruth = { ...new Rect(0.5, 0.5, 0.5, 0.25), classLabel: 0 }
        const lossFunction = createFakeLossFunction(numCells, [groundTruth], [], { anchors })
        const buf = lossFunction.outputTensor.buffer()

        const dx = 2 + inverseSigmoid(0.5)
        const dy = 2 + inverseSigmoid(0.25)
        buf.set(dx, 0, 1, 1, 0  + lossFunction.boxEncodingSize)
        buf.set(dy, 0, 1, 1, 1  + lossFunction.boxEncodingSize)

        expect(Math.abs(tf.sum(lossFunction.computeCoordBoxOffsetError()).dataSync()[0])).toBeGreaterThan(0)

      }))

    })

  })


  describe('computeCoordBoxSizeError', () => {

    describe('error should be zero', () => {

      it('single box at grid (0, 0, 0)', () => tf.tidy(() => {

        const numCells = 2
        const anchor = { x: 2, y: 4 }
        const groundTruth = { ...new Rect(0, 0, 0.5, 0.5), classLabel: 0 }
        const lossFunction = createFakeLossFunction(numCells, [groundTruth], [], { anchors: [anchor]})
        const buf = lossFunction.outputTensor.buffer()

        const dw = Math.log(groundTruth.width / (anchor.x / numCells))
        const dh = Math.log(groundTruth.height / (anchor.y / numCells))
        buf.set(dw, 0, 0, 0, 2)
        buf.set(dh, 0, 0, 0, 3)

        expect(Math.abs(tf.sum(lossFunction.computeCoordBoxSizeError()).dataSync()[0])).toEqual(0)

      }))

      it('single box at grid (1, 1, 0)', () => tf.tidy(() => {

        const numCells = 2
        const anchor = { x: 2, y: 4 }
        const groundTruth = { ...new Rect(0.5, 0.5, 0.5, 0.5), classLabel: 0 }
        const lossFunction = createFakeLossFunction(numCells, [groundTruth], [], { anchors: [anchor]})
        const buf = lossFunction.outputTensor.buffer()

        const dw = Math.log(groundTruth.width / (anchor.x / numCells))
        const dh = Math.log(groundTruth.height / (anchor.y / numCells))
        buf.set(dw, 0, 1, 1, 2)
        buf.set(dh, 0, 1, 1, 3)

        expect(Math.abs(tf.sum(lossFunction.computeCoordBoxSizeError()).dataSync()[0])).toEqual(0)

      }))

      it('single box at grid (1, 1, 1)', () => tf.tidy(() => {

        const numCells = 2
        const anchor = { x: 2, y: 4 }
        const anchors = [{ x: 100, y: 100 }, anchor]
        const groundTruth = { ...new Rect(0.5, 0.5, 0.5, 0.5), classLabel: 0 }
        const lossFunction = createFakeLossFunction(numCells, [groundTruth], [], { anchors })
        const buf = lossFunction.outputTensor.buffer()

        const dw = Math.log(groundTruth.width / (anchor.x / numCells))
        const dh = Math.log(groundTruth.height / (anchor.y / numCells))
        buf.set(dw, 0, 1, 1, 2 + lossFunction.boxEncodingSize)
        buf.set(dh, 0, 1, 1, 3 + lossFunction.boxEncodingSize)

        expect(Math.abs(tf.sum(lossFunction.computeCoordBoxSizeError()).dataSync()[0])).toEqual(0)

      }))

    })

    describe('should compute error', () => {

      it('single box at grid (0, 0, 0)', () => tf.tidy(() => {

        const numCells = 2
        const anchor = { x: 2, y: 4 }
        const groundTruth = { ...new Rect(0, 0, 0.5, 0.5), classLabel: 0 }
        const lossFunction = createFakeLossFunction(numCells, [groundTruth], [], { anchors: [anchor]})
        const buf = lossFunction.outputTensor.buffer()

        const dw = 2 + Math.log(groundTruth.width / (anchor.x / numCells))
        const dh = 2 + Math.log(groundTruth.height / (anchor.y / numCells))
        buf.set(dw, 0, 0, 0, 2)
        buf.set(dh, 0, 0, 0, 3)

        expect(Math.abs(tf.sum(lossFunction.computeCoordBoxSizeError()).dataSync()[0])).toBeGreaterThan(0)

      }))

      it('single box at grid (1, 1, 0)', () => tf.tidy(() => {

        const numCells = 2
        const anchor = { x: 2, y: 4 }
        const groundTruth = { ...new Rect(0.5, 0.5, 0.5, 0.5), classLabel: 0 }
        const lossFunction = createFakeLossFunction(numCells, [groundTruth], [], { anchors: [anchor]})
        const buf = lossFunction.outputTensor.buffer()

        const dw = 2 + Math.log(groundTruth.width / (anchor.x / numCells))
        const dh = 2 + Math.log(groundTruth.height / (anchor.y / numCells))
        buf.set(dw, 0, 1, 1, 2)
        buf.set(dh, 0, 1, 1, 3)

        expect(Math.abs(tf.sum(lossFunction.computeCoordBoxSizeError()).dataSync()[0])).toBeGreaterThan(0)

      }))

      it('single box at grid (1, 1, 1)', () => tf.tidy(() => {

        const numCells = 2
        const anchor = { x: 2, y: 4 }
        const anchors = [{ x: 100, y: 100 }, anchor]
        const groundTruth = { ...new Rect(0.5, 0.5, 0.5, 0.5), classLabel: 0 }
        const lossFunction = createFakeLossFunction(numCells, [groundTruth], [], { anchors })
        const buf = lossFunction.outputTensor.buffer()

        const dw = 2 + Math.log(groundTruth.width / (anchor.x / numCells))
        const dh = 2 + Math.log(groundTruth.height / (anchor.y / numCells))
        buf.set(dw, 0, 1, 1, 2 + lossFunction.boxEncodingSize)
        buf.set(dh, 0, 1, 1, 3 + lossFunction.boxEncodingSize)

        expect(Math.abs(tf.sum(lossFunction.computeCoordBoxSizeError()).dataSync()[0])).toBeGreaterThan(0)

      }))

    })


  })


  // TODO: fix coordinate loss
  xdescribe('computeCoordLoss', () => {

    describe('exact coordinate match', () => {

      it('single box at grid (0, 0), center offset = 0.5', () => tf.tidy(() => {

        const numCells = 2
        const groundTruth = { ...new Rect(0, 0, 0.5, 0.5), classLabel: 0 }
        const lossFunction = createFakeLossFunction(numCells, [groundTruth], [], { anchors: [{ x: 1, y: 1 }]})
        const buf = lossFunction.outputTensor.buffer()

        buf.set(inverseSigmoid(0.5), 0, 0, 0, 0)
        buf.set(inverseSigmoid(0.5), 0, 0, 0, 1)
        buf.set(Math.log(numCells * groundTruth.width), 0, 0, 0, 2)
        buf.set(Math.log(numCells * groundTruth.height), 0, 0, 0, 3)

        expect(lossFunction.computeCoordLoss().dataSync()[0]).toEqual(0)

      }))

      it('single box at grid (1, 1), center offset = 0.5', () => tf.tidy(() => {

        const numCells = 2
        const groundTruth = { ...new Rect(0.5, 0.5, 0.5, 0.5), classLabel: 0 }
        const lossFunction = createFakeLossFunction(numCells, [groundTruth], [], { anchors: [{ x: 1, y: 1 }]})
        const buf = lossFunction.outputTensor.buffer()

        buf.set(inverseSigmoid(0.5), 0, 1, 1, 0)
        buf.set(inverseSigmoid(0.5), 0, 1, 1, 1)
        buf.set(Math.log(numCells * groundTruth.width), 0, 1, 1, 2)
        buf.set(Math.log(numCells * groundTruth.height), 0, 1, 1, 3)

        expect(lossFunction.computeCoordLoss().dataSync()[0]).toEqual(0)

      }))

      it('single box at grid (0, 0), center offset almost 1', () => tf.tidy(() => {

        const numCells = 2
        const groundTruth = { ...new Rect(0.25, 0.25, 0.49, 0.49), classLabel: 0 }
        const lossFunction = createFakeLossFunction(numCells, [groundTruth], [], { anchors: [{ x: 1, y: 1 }]})
        const buf = lossFunction.outputTensor.buffer()

        buf.set(inverseSigmoid(0.99), 0, 0, 0, 0)
        buf.set(inverseSigmoid(0.99), 0, 0, 0, 1)
        buf.set(Math.log(numCells * groundTruth.width), 0, 0, 0, 2)
        buf.set(Math.log(numCells * groundTruth.height), 0, 0, 0, 3)

        expect(lossFunction.computeCoordLoss().dataSync()[0]).toBeCloseTo(0)

      }))

      it('single box at grid (1, 1), center offset almost 0', () => tf.tidy(() => {

        const numCells = 2
        const groundTruth = { ...new Rect(0.25, 0.25, 0.51, 0.51), classLabel: 0 }
        const lossFunction = createFakeLossFunction(numCells, [groundTruth], [], { anchors: [{ x: 1, y: 1 }]})
        const buf = lossFunction.outputTensor.buffer()

        buf.set(inverseSigmoid(0.01), 0, 1, 1, 0)
        buf.set(inverseSigmoid(0.01), 0, 1, 1, 1)
        buf.set(Math.log(numCells * groundTruth.width), 0, 1, 1, 2)
        buf.set(Math.log(numCells * groundTruth.height), 0, 1, 1, 3)

        expect(lossFunction.computeCoordLoss().dataSync()[0]).toBeCloseTo(0)

      }))

    })

  })


})
