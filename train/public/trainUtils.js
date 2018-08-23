const log = (str, ...args) => console.log(`[${[(new Date()).toTimeString().substr(0, 8)]}] ${str || ''}`, ...args)

// https://stackoverflow.com/questions/6274339/how-can-i-shuffle-an-array
function shuffle(a) {
  var j, x, i;
  for (i = a.length - 1; i > 0; i--) {
      j = Math.floor(Math.random() * (i + 1));
      x = a[i];
      a[i] = a[j];
      a[j] = x;
  }
  return a;
}

function saveWeights(net, filename = 'train_tmp') {
  const binaryWeights = new Float32Array(
    net.getParamList()
      .map(({ tensor }) => Array.from(tensor.dataSync()))
      .reduce((flat, arr) => flat.concat(arr))
  )
  saveAs(new Blob([binaryWeights]), filename)
}

function imageToSquare(img, inputSize) {
  const scale = inputSize / Math.max(img.height, img.width)
  const width = scale * img.width
  const height = scale * img.height

  const canvas1 = yolo.createCanvasFromMedia(img)
  const targetCanvas = yolo.createCanvas({ width: inputSize, height: inputSize })
  targetCanvas.getContext('2d').putImageData(canvas1.getContext('2d').getImageData(0, 0, width, height), 0, 0)
  return targetCanvas
}

function getReshapedSize(img, inputSize) {
  const [h, w] = [img.height, img.width]
  const maxDim = Math.max(h, w)

  const f = inputSize / maxDim
  return {
    height: Math.round(h * f),
    width: Math.round(w * f)
  }
}