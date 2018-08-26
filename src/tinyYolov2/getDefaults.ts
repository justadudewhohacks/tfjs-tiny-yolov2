import { SizeType, TinyYolov2ForwardParams, TinyYolov2BackwardOptions } from './types';
import { CELL_SIZE } from './const';

export function getDefaultForwardParams(params: TinyYolov2ForwardParams) {
  return Object.assign(
    {},
    {
      inputSize: SizeType.MD,
      scoreThreshold: 0.5
    },
    params
  )
}

export function getDefaultBackwardOptions(options: TinyYolov2BackwardOptions) {
  return Object.assign(
    {},
    {
      minBoxSize: CELL_SIZE
    },
    options
  )
}