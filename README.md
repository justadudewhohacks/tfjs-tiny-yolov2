# tfjs-tiny-yolov2

[![Build Status](https://travis-ci.org/justadudewhohacks/tfjs-tiny-yolov2.svg?branch=master)](https://travis-ci.org/justadudewhohacks/tfjs-tiny-yolov2)

**JavaScript object detection in the browser based on a tensorflow.js implementation of tiny yolov2.**

Table of Contents:

* **[Pre Trained Models](#pre-trained-models)**
* **[Running the Examples](#running-the-examples)**
* **[Usage](#usage)**
* **[Training your own Object Detector](#training-your-own-object-detector)**
  * **[Defining your Model Config](#defining-your-model-config)**
  * **[Labeling your Data with Ground Truth Boxes](#labeling-your-data-with-ground-truth-boxes)**
  * **[Computing Box Anchors](#computing-box-anchors)**
  * **[Yolo Loss Function](#yolo-loss-function)**
  * **[Initializing the Model Weights](#initializing-the-model-weights)**
  * **[Start Training](#start-training)**
  * **[Overfit first!](#overfit-first)**

<a name="pre-trained-models"></a>

## Pre Trained Models

The VOC and COCO models correspond to the quantized weights from the official [darknet](https://github.com/pjreddie/darknet) repo. The face detector uses depthwise separable convolutions instead of regular convolutions allowing for much faster prediction and a tiny model size, which is well suited for object detection on mobile devices as well. I trained the face detection model from scratch. Have a look at the **[Training your own Object Detector](#training-your-own-object-detector)** section if you want to train such a model for your own dataset!

### Pascal VOC

![voc1](https://user-images.githubusercontent.com/31125521/44733258-7ca2a000-aae7-11e8-93f3-07be9943e222.jpg)
![voc2](https://user-images.githubusercontent.com/31125521/44733259-7d3b3680-aae7-11e8-9794-ab00edef9a48.jpg)

### COCO

![coco1](https://user-images.githubusercontent.com/31125521/44733254-7ca2a000-aae7-11e8-9113-28eaea552093.jpg)
![coco2](https://user-images.githubusercontent.com/31125521/44733256-7ca2a000-aae7-11e8-98f6-26853a12248a.jpg)

### Face Detection

The face detection model is one of the models available in [face-api.js](https://github.com/justadudewhohacks/face-api.js).

![face](https://user-images.githubusercontent.com/31125521/44733257-7ca2a000-aae7-11e8-9ede-1a38f20e36be.jpg)

<a name="running-the-examples"></a>

## Running the Examples

``` bash
cd examples
npm i
npm start
```

Browse to http://localhost:3000/.

<a name="usage"></a>

## Usage

Get the latest build from dist/tiny-yolov2.js or dist/tiny-yolov2.min.js and include the script:

``` html
<script src="tiny-yolov2.js"></script>
```

Simply load the model:

``` javascript
const config = // yolo config
const net = new yolo.TinyYolov2(config)
await net.load(`voc_model-weights_manifest.json`)
```

The config file of the VOC model looks as follows:

``` js
{
  // the pre trained VOC model uses regular convolutions
  "withSeparableConvs": false,
  // iou threshold for nonMaxSuppression
  "iouThreshold": 0.4,
  // anchor box dimensions, relative to cell size (32px)
  "anchors": [
    { "x": 1.08, "y": 1.19 },
    { "x": 3.42, "y": 4.41 },
    { "x": 6.63, "y": 11.38 },
    { "x": 9.42, "y": 5.11 },
    { "x": 16.62, "y": 10.52 }
  ],
  // class labels in correct order
  "classes": [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
  ]
}
```

Inference and drawing the results:

``` javascript
const forwardParams = {
  inputSize: 416,
  scoreThreshold: 0.8
}

const detections = await net.detect('myInputImage', forwardParams)
yolo.drawDetection('myCanvas', detections)
```

Also check out the examples.

<a name="training-your-own-object-detector"></a>

## Training your own Object Detector

If you want to train your own object detector, I would suggest training a model using separable convolutions, as it will allow for much faster inference times and the training process will converge much faster, as there are significantly less parameters to train.

Training a multiclass detector will take quite some time, depending on how much classes you are training your object detector on. However, training a single class detector it is possible to get already pretty good results after training for only a few epochs.

<a name="defining-your-model-config"></a>

### Defining your Model Config

``` js
{
  // use separable convolutions over regular convolutions
  "withSeparableConvs": true,
  // iou threshold for nonMaxSuppression
  "iouThreshold": 0.4,
  // instructions for how to determine anchors is given below
  "anchors": [...],
  // whatever kind of objects you are training your object detector on
  "classes": ["cat"],
  // optionally you can compute the mean RGB value for your dataset and
  // pass it in the config for performing mean value subtraction on your
  // input images
  "meanRgb": [...],
  // scale factors for each loss term (only required for training),
  // explained below
  "objectScale": 5,
  "noObjectScale": 1,
  "coordScale": 1,
  "classScale": 1
}
```

<a name="labeling-your-data-with-ground-truth-boxes"></a>

### Labeling your Data with Ground Truth Boxes

For each image in your training set, you should create a corresponding json file, containing the bounding boxes and class labels of each of the instance of objects located in that image. The bounding box dimensions should be relative to the image dimensions.

Consider an image with a width and height of 400px, showing a single cat, which is spanned by the bounding box at x = 50px, y = 100px (upper left corner) with a box size of width = 200px and height = 100px. The corresponding json file should look as follows (note, it is an array of all bounding boxes for that image):

``` json
[
  {
    "x": 0.125,
    "y": 0.25,
    "width": 0.5,
    "height": 0.25,
    "label": "cat"
  }
]
```

<a name="computing-box-anchors"></a>

### Computing Box Anchors

Before training your detector, you want to compute 5 anchor boxes over your training set. An anchor box is basically an object of shape { "x": boxWidth / 32, "y": boxHeight / 32 } where x and y are the anchor box sizes relative to the grid cell size (32px).

To determine the 5 anchor boxes, you want to simply perform kmeans clustering with 5 clusters over the width and height of each ground truth box of your training set. There should be plenty of options out there, which you can use for kmeans clustering, but I will provide a script for that, coming soon...

<a name="yolo-loss-function"></a>

### Yolo Loss Function

The Yolo loss function computes the sum of the coordinate, object, class and no object loss. You can tune the weight of each loss term contributing to the totoal loss by adjusting the corresponding scale parameters in your config file, as mentioned above.

The no object loss term penalizes the scores of the bounding box of all the box anchors in the grid, which do not have a corresponding ground truth bounding box. In other words, they should optimally predict a score of 0, if there is no object of interest at that position.

On the other hand, the object, class and coordinate loss terms refer to the accuracy of the prediction at each anchor position where there is a ground truth bounding box. The coordinate loss simply penalizes the difference between predicted bounding box coordinates and ground truth box coordinates, the object loss penalizes the difference of the predicted confidence score to the box IOU.

The class loss penalizes the confidence score of the predicted score. Note, that training a single class object detector you can simply ignore that parameter, as the class loss is always 0 in that case.

PS: You can simply go with the default values in the above shown config example.

<a name="initializing-the-model-weights"></a>

### Initializing the Model Weights

Training a model from scratch, you need some weights to begin with. Simply open initWeights.html located in the /train folder of the repo in your browser. Enter the number of classes, hit save and use the saved file as the initial checkpoint weight file.

<a name="start-training"></a>

### Start Training

For a complete example, also check out the /train folder at the root of this repo, which also contains some tooling to save intermediary checkpoints of your model weights as well as statistics of the average loss after each epoch.

Set up the model for training:

``` javascript
const config = // your config

// simply use any of the optimizer provided by tfjs (I usually use adam)
const learningRate = 0.001
const optimizer = tf.train.adam(learningRate, 0.9, 0.999, 1e-8)

// initialize a trainable TinyYolov2
const net = new yolo.TinyYolov2Trainable(config, optimizer)

// load initial weights or the weights of any checkpoint
const checkpointUri = 'checkpoints/initial_glorot_1_classes.weights'
const weights = new Float32Array(await (await fetch(checkpointUri)).arrayBuffer())
await net.load(weights)
```

What I usually do is naming the json files the same as the corresponding image, e.g. *img1.jpg* and *img1.json* and provide an endpoint to retrieve the json file names as an array:

``` javascript
const boxJsonUris = (await fetch('/boxJsonUris')).json()
```

Furthermore you can choose to train your model on a fixed input size or you can perform multi scale training, which is a good way to improve the accuracy of your model at different scales. This can also be helpful to augment your data, in case you only have a limited number of training samples:

``` javascript
// should be multiples of 32 (grid cell size)
const trainingSizes = [160, 224, 320, 416]
```

Then we can actually train it:

``` javascript
for (let epoch = startEpoch; epoch < maxEpoch; epoch++) {

  // always shuffle your inputs for each epoch
  const shuffledInputs = yolo.shuffleArray(boxJsonUris)

  // loop through shuffled inputs
  for (let dataIdx = 0; dataIdx < shuffledInputs.length; dataIdx++) {

    // fetch image and corresponding ground truth bounding boxes
    const boxJsonUri = shuffledInputs[dataIdx]
    const imgUri = boxJsonUri.replace('.json', '.jpg')

    const groundTruth = await (await fetch(boxJsonUri)).json()
    const img = await yolo.bufferToImage(await (await fetch(imgUri)).blob())

    // rescale and backward pass input image for each input size
    for (let sizeIdx = 0; sizeIdx < trainSizes.length; sizeIdx++) {

      const inputSize = trainSizes[sizeIdx]

      const backwardOptions = {
        // filter boxes with width < 32 or height < 32
        minBoxSize: 32,
        // log computed losses
        reportLosses: function({ losses, numBoxes, inputSize }) {
          console.log(`ground truth boxes: ${numBoxes} (${inputSize})`)
          console.log(`noObjectLoss[${dataIdx}]: ${yolo.round(losses.noObjectLoss, 4)}`)
          console.log(`objectLoss[${dataIdx}]: ${yolo.round(losses.objectLoss, 4)}`)
          console.log(`coordLoss[${dataIdx}]: ${yolo.round(losses.coordLoss, 4)}`)
          console.log(`classLoss[${dataIdx}]: ${yolo.round(losses.classLoss, 4)}`)
          console.log(`totalLoss[${dataIdx}]: ${yolo.round(losses.totalLoss, 4)}`)
        }
      }

      const loss = await net.backward(img, groundTruth, inputSize, backwardOptions)

      if (loss) {
        // don't forget to free the loss tensor
        loss.dispose()
      } else {
        console.log('no boxes remaining after filtering')
      }

    }
  }
}
```

<a name="overfit-first"></a>

### Overfit first!

Generally it's a good idea, to overfit on a small subset of your training data, to verify, that the loss is converging and that your detector is actually learning something. Therefore, you can simply train your detector on 10 - 20 images of your training data for some epochs. Once the loss converges, save the model, run inference on these 10 - 20 images to view the predicted bounding boxes and compare them to the ground truth boxes.