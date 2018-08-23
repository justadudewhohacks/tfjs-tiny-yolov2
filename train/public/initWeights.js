function toDataArray(tensor) {
  return Array.from(tensor.dataSync())
}

function flatten(arrs) {
  return arrs.reduce((flat, arr) => flat.concat(arr))
}

function initTinyYolov2SeparableWeights(
  numClasses,
  numAnchors = 5,
  initializer = { apply: (shape) => tf.randomNormal(shape) }
) {

  if (typeof numClasses !== 'number') {
    throw new Error(`initTinyYolov2SeparableWeights - invalid number of classes: ${numClasses}`)
  }

  function initSeparableConvWeights(inChannels, outChannels) {
    return flatten(
      [
        // depthwise filter
        initializer.apply([3, 3, inChannels, 1]),
        // pointwise filter
        initializer.apply([1, 1, inChannels, outChannels]),
        // bias
        tf.zeros([outChannels])
      ]
        .map(toDataArray)
    )
  }

  const outDepth = numAnchors * (5 + numClasses)
  const separableConvWeights = flatten([
    initSeparableConvWeights(3, 16),
    initSeparableConvWeights(16, 32),
    initSeparableConvWeights(32, 64),
    initSeparableConvWeights(64, 128),
    initSeparableConvWeights(128, 256),
    initSeparableConvWeights(256, 512),
    initSeparableConvWeights(512, 1024),
    initSeparableConvWeights(1024, 1024),
  ])

  const outConv = flatten(
    [
      initializer.apply([1, 1, 1024, outDepth]),
      tf.zeros([outDepth])
    ]
      .map(toDataArray)
  )
  return new Float32Array(separableConvWeights.concat(outConv))
}