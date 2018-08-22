import * as tf from '@tensorflow/tfjs-core';

import { iou } from '../src/iou';
import { Rect } from '../src/Rect';


describe('iou', () => {

  it('should be 1.0', () => tf.tidy(() => {

    const box = new Rect(0, 0, 20, 20).toBoundingBox()

    expect(iou(box, box)).toEqual(1)

  }))

  it('should be 0', () => tf.tidy(() => {

    const box1 = new Rect(0, 0, 20, 20).toBoundingBox()
    const box2 = new Rect(20, 20, 20, 20).toBoundingBox()

    expect(iou(box1, box2)).toEqual(0)

  }))

  it('should be 0.5', () => tf.tidy(() => {

    const box1 = new Rect(0, 0, 20, 20).toBoundingBox()
    const box2 = new Rect(0, 0, 10, 20).toBoundingBox()

    expect(iou(box1, box2)).toEqual(0.5)

  }))

  it('should be 0.5', () => tf.tidy(() => {

    const box1 = new Rect(0, 0, 20, 20).toBoundingBox()
    const box2 = new Rect(0, 10, 20, 10).toBoundingBox()

    expect(iou(box1, box2)).toEqual(0.5)

  }))

})