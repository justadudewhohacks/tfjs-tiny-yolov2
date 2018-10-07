import { CELL_SIZE } from './const';
import { TinyYolov2BackwardOptions } from './types';

export function getDefaultBackwardOptions(options: TinyYolov2BackwardOptions) {
  return Object.assign(
    {},
    {
      minBoxSize: CELL_SIZE
    },
    options
  )
}