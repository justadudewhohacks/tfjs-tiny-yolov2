function randomCrop(img, rects) {

  if (!rects.length) {
    throw new Error('randomCrop - rects empty')
  }

  let minX = img.width
  let minY = img.height
  let maxX = 0
  let maxY = 0

  const rescaledRects = rects.map(({ x, y, width, height }) =>
    new yolo.Rect(x * img.width, y * img.height, width * img.width, height * img.height)
  )

  rescaledRects.forEach(({ x, y, width, height }) => {
    minX = Math.min(minX, x)
    minY = Math.min(minY, y)
    maxX = Math.max(maxX, x + width)
    maxY = Math.max(maxY, y + height)
  })

  console.log(rects)
  console.log(minX, minY, maxX, maxY)

  const x0 = Math.random() * minX
  const y0 = Math.random() * minY

  const x1 = (Math.random() * Math.abs(img.width - maxX)) + maxX
  const y1 = (Math.random() * Math.abs(img.height - maxY)) + maxY

  console.log(x0, y0, x1, y1)

  const targetCanvas = yolo.createCanvas({ width: x1 - x0, height: y1 - y0 })
  const inputCanvas = img instanceof HTMLCanvasElement ? img : yolo.createCanvasFromMedia(img)
  const region = yolo.getContext2dOrThrow(inputCanvas)
    .getImageData(x0, y0, targetCanvas.width, targetCanvas.height)
    yolo.getContext2dOrThrow(targetCanvas).putImageData(region, 0, 0)

  return {
    croppedImage: targetCanvas,
    shiftedRects: rescaledRects.map(rect =>
      new yolo.Rect(rect.x - x0, rect.y - y0, rect.width, rect.height)
        .rescale({ width: 1 / targetCanvas.width, height: 1 / targetCanvas.height })
    )
  }
}