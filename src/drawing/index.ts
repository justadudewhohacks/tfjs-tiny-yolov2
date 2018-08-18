import { ObjectDetection } from '../ObjectDetection';
import { getContext2dOrThrow, resolveInput, round } from '../utils';
import { DrawBoxOptions, DrawOptions, DrawTextOptions } from './types';

export function getDefaultDrawOptions(): DrawOptions {
  return {
    color: 'blue',
    lineWidth: 2,
    fontSize: 20,
    fontStyle: 'Georgia'
  }
}

export function drawBox(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  options: DrawBoxOptions
) {
  const drawOptions = Object.assign(
    getDefaultDrawOptions(),
    (options || {})
  )

  ctx.strokeStyle = drawOptions.color
  ctx.lineWidth = drawOptions.lineWidth
  ctx.strokeRect(x, y, w, h)
}

export function drawText(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  text: string,
  options: DrawTextOptions
) {
  const drawOptions = Object.assign(
    getDefaultDrawOptions(),
    (options || {})
  )

  const padText = 2 + drawOptions.lineWidth

  ctx.fillStyle = drawOptions.color
  ctx.font = `${drawOptions.fontSize}px ${drawOptions.fontStyle}`
  ctx.fillText(text, x + padText, y + padText + (drawOptions.fontSize * 0.6))
}

export function drawDetection(
  canvasArg: string | HTMLCanvasElement,
  detection: ObjectDetection | ObjectDetection[],
  options?: DrawBoxOptions & DrawTextOptions & { withScore: boolean }
) {
  const canvas = resolveInput(canvasArg)
  if (!(canvas instanceof HTMLCanvasElement)) {
    throw new Error('drawBox - expected canvas to be of type: HTMLCanvasElement')
  }

  const detectionArray = Array.isArray(detection)
    ? detection
    : [detection]

  detectionArray.forEach((det) => {
    const {
      x,
      y,
      width,
      height
    } = det.getBox()

    const drawOptions = Object.assign(
      getDefaultDrawOptions(),
      (options || {})
    )

    const { withScore } = Object.assign({ withScore: true }, (options || {}))

    const ctx = getContext2dOrThrow(canvas)
    drawBox(
      ctx,
      x,
      y,
      width,
      height,
      drawOptions
    )
    if (withScore) {
      drawText(
        ctx,
        x,
        y,
        `${round(det.getScore())}`,
        drawOptions
      )
    }
  })
}