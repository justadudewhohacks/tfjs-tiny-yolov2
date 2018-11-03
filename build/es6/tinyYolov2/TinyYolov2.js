import * as tslib_1 from "tslib";
import * as tf from '@tensorflow/tfjs-core';
import { BoundingBox, NeuralNetwork, nonMaxSuppression, normalize, ObjectDetection, sigmoid, toNetInput, } from 'tfjs-image-recognition-base';
import { convLayer } from '../common';
import { validateConfig } from './config';
import { DEFAULT_FILTER_SIZES } from './const';
import { convWithBatchNorm } from './convWithBatchNorm';
import { depthwiseSeparableConv } from './depthwiseSeparableConv';
import { extractParams } from './extractParams';
import { extractParamsFromWeigthMap } from './extractParamsFromWeigthMap';
import { leaky } from './leaky';
import { TinyYolov2Options } from './TinyYolov2Options';
var TinyYolov2 = /** @class */ (function (_super) {
    tslib_1.__extends(TinyYolov2, _super);
    function TinyYolov2(config) {
        var _this = _super.call(this, 'TinyYolov2') || this;
        validateConfig(config);
        _this._config = config;
        return _this;
    }
    Object.defineProperty(TinyYolov2.prototype, "config", {
        get: function () {
            return this._config;
        },
        enumerable: true,
        configurable: true
    });
    Object.defineProperty(TinyYolov2.prototype, "withClassScores", {
        get: function () {
            return this.config.withClassScores || this.config.classes.length > 1;
        },
        enumerable: true,
        configurable: true
    });
    Object.defineProperty(TinyYolov2.prototype, "boxEncodingSize", {
        get: function () {
            return 5 + (this.withClassScores ? this.config.classes.length : 0);
        },
        enumerable: true,
        configurable: true
    });
    TinyYolov2.prototype.runTinyYolov2 = function (x, params) {
        var out = convWithBatchNorm(x, params.conv0);
        out = tf.maxPool(out, [2, 2], [2, 2], 'same');
        out = convWithBatchNorm(out, params.conv1);
        out = tf.maxPool(out, [2, 2], [2, 2], 'same');
        out = convWithBatchNorm(out, params.conv2);
        out = tf.maxPool(out, [2, 2], [2, 2], 'same');
        out = convWithBatchNorm(out, params.conv3);
        out = tf.maxPool(out, [2, 2], [2, 2], 'same');
        out = convWithBatchNorm(out, params.conv4);
        out = tf.maxPool(out, [2, 2], [2, 2], 'same');
        out = convWithBatchNorm(out, params.conv5);
        out = tf.maxPool(out, [2, 2], [1, 1], 'same');
        out = convWithBatchNorm(out, params.conv6);
        out = convWithBatchNorm(out, params.conv7);
        return convLayer(out, params.conv8, 'valid', false);
    };
    TinyYolov2.prototype.runMobilenet = function (x, params) {
        var out = this.config.isFirstLayerConv2d
            ? leaky(convLayer(x, params.conv0, 'valid', false))
            : depthwiseSeparableConv(x, params.conv0);
        out = tf.maxPool(out, [2, 2], [2, 2], 'same');
        out = depthwiseSeparableConv(out, params.conv1);
        out = tf.maxPool(out, [2, 2], [2, 2], 'same');
        out = depthwiseSeparableConv(out, params.conv2);
        out = tf.maxPool(out, [2, 2], [2, 2], 'same');
        out = depthwiseSeparableConv(out, params.conv3);
        out = tf.maxPool(out, [2, 2], [2, 2], 'same');
        out = depthwiseSeparableConv(out, params.conv4);
        out = tf.maxPool(out, [2, 2], [2, 2], 'same');
        out = depthwiseSeparableConv(out, params.conv5);
        out = tf.maxPool(out, [2, 2], [1, 1], 'same');
        out = params.conv6 ? depthwiseSeparableConv(out, params.conv6) : out;
        out = params.conv7 ? depthwiseSeparableConv(out, params.conv7) : out;
        return convLayer(out, params.conv8, 'valid', false);
    };
    TinyYolov2.prototype.forwardInput = function (input, inputSize) {
        var _this = this;
        var params = this.params;
        if (!params) {
            throw new Error('TinyYolov2 - load model before inference');
        }
        return tf.tidy(function () {
            var batchTensor = input.toBatchTensor(inputSize, false).toFloat();
            batchTensor = _this.config.meanRgb
                ? normalize(batchTensor, _this.config.meanRgb)
                : batchTensor;
            batchTensor = batchTensor.div(tf.scalar(256));
            return _this.config.withSeparableConvs
                ? _this.runMobilenet(batchTensor, params)
                : _this.runTinyYolov2(batchTensor, params);
        });
    };
    TinyYolov2.prototype.forward = function (input, inputSize) {
        return tslib_1.__awaiter(this, void 0, void 0, function () {
            var _a;
            return tslib_1.__generator(this, function (_b) {
                switch (_b.label) {
                    case 0:
                        _a = this.forwardInput;
                        return [4 /*yield*/, toNetInput(input)];
                    case 1: return [4 /*yield*/, _a.apply(this, [_b.sent(), inputSize])];
                    case 2: return [2 /*return*/, _b.sent()];
                }
            });
        });
    };
    TinyYolov2.prototype.detect = function (input, forwardParams) {
        if (forwardParams === void 0) { forwardParams = {}; }
        return tslib_1.__awaiter(this, void 0, void 0, function () {
            var _this = this;
            var _a, inputSize, scoreThreshold, netInput, out, out0, inputDimensions, results, boxes, scores, classScores, classNames, indices, detections;
            return tslib_1.__generator(this, function (_b) {
                switch (_b.label) {
                    case 0:
                        _a = new TinyYolov2Options(forwardParams), inputSize = _a.inputSize, scoreThreshold = _a.scoreThreshold;
                        return [4 /*yield*/, toNetInput(input)];
                    case 1:
                        netInput = _b.sent();
                        return [4 /*yield*/, this.forwardInput(netInput, inputSize)];
                    case 2:
                        out = _b.sent();
                        out0 = tf.tidy(function () { return tf.unstack(out)[0].expandDims(); });
                        inputDimensions = {
                            width: netInput.getInputWidth(0),
                            height: netInput.getInputHeight(0)
                        };
                        results = this.extractBoxes(out0, netInput.getReshapedInputDimensions(0), scoreThreshold);
                        out.dispose();
                        out0.dispose();
                        boxes = results.map(function (res) { return res.box; });
                        scores = results.map(function (res) { return res.score; });
                        classScores = results.map(function (res) { return res.classScore; });
                        classNames = results.map(function (res) { return _this.config.classes[res.label]; });
                        indices = nonMaxSuppression(boxes.map(function (box) { return box.rescale(inputSize); }), scores, this.config.iouThreshold, true);
                        detections = indices.map(function (idx) {
                            return new ObjectDetection(scores[idx], classScores[idx], classNames[idx], boxes[idx], inputDimensions);
                        });
                        return [2 /*return*/, detections];
                }
            });
        });
    };
    TinyYolov2.prototype.getDefaultModelName = function () {
        return '';
    };
    TinyYolov2.prototype.extractParamsFromWeigthMap = function (weightMap) {
        return extractParamsFromWeigthMap(weightMap, this.config);
    };
    TinyYolov2.prototype.extractParams = function (weights) {
        var filterSizes = this.config.filterSizes || DEFAULT_FILTER_SIZES;
        var numFilters = filterSizes ? filterSizes.length : undefined;
        if (numFilters !== 7 && numFilters !== 8 && numFilters !== 9) {
            throw new Error("TinyYolov2 - expected 7 | 8 | 9 convolutional filters, but found " + numFilters + " filterSizes in config");
        }
        return extractParams(weights, this.config, this.boxEncodingSize, filterSizes);
    };
    TinyYolov2.prototype.extractBoxes = function (outputTensor, inputBlobDimensions, scoreThreshold) {
        var _this = this;
        var width = inputBlobDimensions.width, height = inputBlobDimensions.height;
        var inputSize = Math.max(width, height);
        var correctionFactorX = inputSize / width;
        var correctionFactorY = inputSize / height;
        var numCells = outputTensor.shape[1];
        var numBoxes = this.config.anchors.length;
        var _a = tf.tidy(function () {
            var reshaped = outputTensor.reshape([numCells, numCells, numBoxes, _this.boxEncodingSize]);
            var boxes = reshaped.slice([0, 0, 0, 0], [numCells, numCells, numBoxes, 4]);
            var scores = reshaped.slice([0, 0, 0, 4], [numCells, numCells, numBoxes, 1]);
            var classScores = _this.withClassScores
                ? tf.softmax(reshaped.slice([0, 0, 0, 5], [numCells, numCells, numBoxes, _this.config.classes.length]), 3)
                : tf.scalar(0);
            return [boxes, scores, classScores];
        }), boxesTensor = _a[0], scoresTensor = _a[1], classScoresTensor = _a[2];
        var results = [];
        for (var row = 0; row < numCells; row++) {
            for (var col = 0; col < numCells; col++) {
                for (var anchor = 0; anchor < numBoxes; anchor++) {
                    var score = sigmoid(scoresTensor.get(row, col, anchor, 0));
                    if (!scoreThreshold || score > scoreThreshold) {
                        var ctX = ((col + sigmoid(boxesTensor.get(row, col, anchor, 0))) / numCells) * correctionFactorX;
                        var ctY = ((row + sigmoid(boxesTensor.get(row, col, anchor, 1))) / numCells) * correctionFactorY;
                        var width_1 = ((Math.exp(boxesTensor.get(row, col, anchor, 2)) * this.config.anchors[anchor].x) / numCells) * correctionFactorX;
                        var height_1 = ((Math.exp(boxesTensor.get(row, col, anchor, 3)) * this.config.anchors[anchor].y) / numCells) * correctionFactorY;
                        var x = (ctX - (width_1 / 2));
                        var y = (ctY - (height_1 / 2));
                        var pos = { row: row, col: col, anchor: anchor };
                        var _b = this.withClassScores
                            ? this.extractPredictedClass(classScoresTensor, pos)
                            : { classScore: 1, label: 0 }, classScore = _b.classScore, label = _b.label;
                        results.push(tslib_1.__assign({ box: new BoundingBox(x, y, x + width_1, y + height_1), score: score, classScore: score * classScore, label: label }, pos));
                    }
                }
            }
        }
        boxesTensor.dispose();
        scoresTensor.dispose();
        classScoresTensor.dispose();
        return results;
    };
    TinyYolov2.prototype.extractPredictedClass = function (classesTensor, pos) {
        var row = pos.row, col = pos.col, anchor = pos.anchor;
        return Array(this.config.classes.length).fill(0)
            .map(function (_, i) { return classesTensor.get(row, col, anchor, i); })
            .map(function (classScore, label) { return ({
            classScore: classScore,
            label: label
        }); })
            .reduce(function (max, curr) { return max.classScore > curr.classScore ? max : curr; });
    };
    return TinyYolov2;
}(NeuralNetwork));
export { TinyYolov2 };
//# sourceMappingURL=TinyYolov2.js.map