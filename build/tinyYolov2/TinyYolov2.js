"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tslib_1 = require("tslib");
var tf = require("@tensorflow/tfjs-core");
var BoundingBox_1 = require("../BoundingBox");
var convLayer_1 = require("../commons/convLayer");
var NeuralNetwork_1 = require("../commons/NeuralNetwork");
var nonMaxSuppression_1 = require("../commons/nonMaxSuppression");
var normalize_1 = require("../commons/normalize");
var ObjectDetection_1 = require("../ObjectDetection");
var toNetInput_1 = require("../toNetInput");
var utils_1 = require("../utils");
var config_1 = require("./config");
var const_1 = require("./const");
var convWithBatchNorm_1 = require("./convWithBatchNorm");
var extractParams_1 = require("./extractParams");
var getDefaultParams_1 = require("./getDefaultParams");
var loadQuantizedParams_1 = require("./loadQuantizedParams");
var TinyYolov2 = /** @class */ (function (_super) {
    tslib_1.__extends(TinyYolov2, _super);
    function TinyYolov2(config) {
        var _this = _super.call(this, 'TinyYolov2') || this;
        config_1.validateConfig(config);
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
    TinyYolov2.prototype.forwardInput = function (input, inputSize) {
        var _this = this;
        var params = this.params;
        if (!params) {
            throw new Error('TinyYolov2 - load model before inference');
        }
        var out = tf.tidy(function () {
            var batchTensor = input.toBatchTensor(inputSize, false);
            batchTensor = _this.config.meanRgb
                ? normalize_1.normalize(batchTensor, _this.config.meanRgb)
                : batchTensor;
            batchTensor = batchTensor.div(tf.scalar(256));
            var out = convWithBatchNorm_1.convWithBatchNorm(batchTensor, params.conv0);
            out = tf.maxPool(out, [2, 2], [2, 2], 'same');
            out = convWithBatchNorm_1.convWithBatchNorm(out, params.conv1);
            out = tf.maxPool(out, [2, 2], [2, 2], 'same');
            out = convWithBatchNorm_1.convWithBatchNorm(out, params.conv2);
            out = tf.maxPool(out, [2, 2], [2, 2], 'same');
            out = convWithBatchNorm_1.convWithBatchNorm(out, params.conv3);
            out = tf.maxPool(out, [2, 2], [2, 2], 'same');
            out = convWithBatchNorm_1.convWithBatchNorm(out, params.conv4);
            out = tf.maxPool(out, [2, 2], [2, 2], 'same');
            out = convWithBatchNorm_1.convWithBatchNorm(out, params.conv5);
            out = tf.maxPool(out, [2, 2], [1, 1], 'same');
            out = convWithBatchNorm_1.convWithBatchNorm(out, params.conv6);
            out = convWithBatchNorm_1.convWithBatchNorm(out, params.conv7);
            out = convLayer_1.convLayer(out, params.conv8, 'valid', false);
            return out;
        });
        return out;
    };
    TinyYolov2.prototype.forward = function (input, inputSize) {
        return tslib_1.__awaiter(this, void 0, void 0, function () {
            var _a;
            return tslib_1.__generator(this, function (_b) {
                switch (_b.label) {
                    case 0:
                        _a = this.forwardInput;
                        return [4 /*yield*/, toNetInput_1.toNetInput(input, true, true)];
                    case 1: return [4 /*yield*/, _a.apply(this, [_b.sent(), inputSize])];
                    case 2: return [2 /*return*/, _b.sent()];
                }
            });
        });
    };
    TinyYolov2.prototype.detect = function (input, forwardParams) {
        if (forwardParams === void 0) { forwardParams = {}; }
        return tslib_1.__awaiter(this, void 0, void 0, function () {
            var _a, _inputSize, scoreThreshold, inputSize, netInput, out, out0, inputDimensions, results, boxes, scores, classNames, indices, detections;
            return tslib_1.__generator(this, function (_b) {
                switch (_b.label) {
                    case 0:
                        _a = getDefaultParams_1.getDefaultParams(forwardParams), _inputSize = _a.inputSize, scoreThreshold = _a.scoreThreshold;
                        inputSize = typeof _inputSize === 'string'
                            ? const_1.INPUT_SIZES[_inputSize]
                            : _inputSize;
                        if (typeof inputSize !== 'number') {
                            throw new Error("TinyYolov2 - unknown inputSize: " + inputSize + ", expected number or one of xs | sm | md | lg");
                        }
                        return [4 /*yield*/, toNetInput_1.toNetInput(input, true)];
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
                        results = this.extractBoxes(out0, scoreThreshold, netInput.getReshapedInputDimensions(0));
                        out.dispose();
                        out0.dispose();
                        boxes = results.map(function (res) { return res.box; });
                        scores = results.map(function (res) { return res.score; });
                        classNames = results.map(function (res) { return res.className; });
                        indices = nonMaxSuppression_1.nonMaxSuppression(boxes.map(function (box) { return box.rescale(inputSize); }), scores, this.config.iouThreshold, true);
                        detections = indices.map(function (idx) {
                            return new ObjectDetection_1.ObjectDetection(scores[idx], classNames[idx], boxes[idx].toRect(), inputDimensions);
                        });
                        return [2 /*return*/, detections];
                }
            });
        });
    };
    TinyYolov2.prototype.extractBoxes = function (outputTensor, scoreThreshold, inputBlobDimensions) {
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
            var classes = _this.withClassScores
                ? reshaped.slice([0, 0, 0, 5], [numCells, numCells, numBoxes, _this.config.classes.length])
                : tf.scalar(0);
            return [boxes, scores, classes];
        }), boxesTensor = _a[0], scoresTensor = _a[1], classesTensor = _a[2];
        var results = [];
        for (var row = 0; row < numCells; row++) {
            for (var col = 0; col < numCells; col++) {
                for (var anchor = 0; anchor < numBoxes; anchor++) {
                    var score = utils_1.sigmoid(scoresTensor.get(row, col, anchor, 0));
                    if (!scoreThreshold || score > scoreThreshold) {
                        var ctX = ((col + utils_1.sigmoid(boxesTensor.get(row, col, anchor, 0))) / numCells) * correctionFactorX;
                        var ctY = ((row + utils_1.sigmoid(boxesTensor.get(row, col, anchor, 1))) / numCells) * correctionFactorY;
                        var width_1 = ((Math.exp(boxesTensor.get(row, col, anchor, 2)) * this.config.anchors[anchor].x) / numCells) * correctionFactorX;
                        var height_1 = ((Math.exp(boxesTensor.get(row, col, anchor, 3)) * this.config.anchors[anchor].y) / numCells) * correctionFactorY;
                        var x = (ctX - (width_1 / 2));
                        var y = (ctY - (height_1 / 2));
                        var pos = { row: row, col: col, anchor: anchor };
                        var classScores = this.withClassScores
                            ? this.extractClassScores(classesTensor, score, pos)
                            : [score];
                        var _b = classScores
                            .map(function (classScore, idx) { return ({
                            className: _this.config.classes[idx],
                            classScore: classScore
                        }); })
                            .reduce(function (max, curr) { return max.classScore > curr.classScore ? max : curr; }), classScore = _b.classScore, className = _b.className;
                        results.push(tslib_1.__assign({ box: new BoundingBox_1.BoundingBox(x, y, x + width_1, y + height_1), score: classScore, className: className }, pos));
                    }
                }
            }
        }
        boxesTensor.dispose();
        scoresTensor.dispose();
        return results;
    };
    TinyYolov2.prototype.extractClassScores = function (classesTensor, score, pos) {
        var row = pos.row, col = pos.col, anchor = pos.anchor;
        var classesData = Array(this.config.classes.length).fill(0).map(function (_, i) { return classesTensor.get(row, col, anchor, i); });
        var maxClass = classesData.reduce(function (max, c) { return max > c ? max : c; });
        var classes = classesData.map(function (c) { return Math.exp(c - maxClass); });
        var sum = classes.reduce(function (sum, c) { return sum + c; });
        return classes.map(function (c) { return (c * score) / sum; });
    };
    TinyYolov2.prototype.loadQuantizedParams = function (modelUri) {
        if (!modelUri) {
            throw new Error('loadQuantizedParams - please specify the modelUri');
        }
        return loadQuantizedParams_1.loadQuantizedParams(modelUri, this.config.withSeparableConvs);
    };
    TinyYolov2.prototype.extractParams = function (weights) {
        return extractParams_1.extractParams(weights, this.config.withSeparableConvs, this.boxEncodingSize);
    };
    return TinyYolov2;
}(NeuralNetwork_1.NeuralNetwork));
exports.TinyYolov2 = TinyYolov2;
//# sourceMappingURL=TinyYolov2.js.map