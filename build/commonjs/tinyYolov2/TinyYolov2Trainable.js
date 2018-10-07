"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tslib_1 = require("tslib");
var tf = require("@tensorflow/tfjs-core");
var tfjs_image_recognition_base_1 = require("tfjs-image-recognition-base");
var config_1 = require("./config");
var getDefaultBackwardOptions_1 = require("./getDefaultBackwardOptions");
var TinyYolov2_1 = require("./TinyYolov2");
var TinyYolov2LossFunction_1 = require("./TinyYolov2LossFunction");
var TinyYolov2Trainable = /** @class */ (function (_super) {
    tslib_1.__extends(TinyYolov2Trainable, _super);
    function TinyYolov2Trainable(trainableConfig, optimizer) {
        var _this = _super.call(this, trainableConfig) || this;
        _this._trainableConfig = config_1.validateTrainConfig(trainableConfig);
        _this._optimizer = optimizer;
        return _this;
    }
    Object.defineProperty(TinyYolov2Trainable.prototype, "trainableConfig", {
        get: function () {
            return this._trainableConfig;
        },
        enumerable: true,
        configurable: true
    });
    Object.defineProperty(TinyYolov2Trainable.prototype, "optimizer", {
        get: function () {
            return this._optimizer;
        },
        enumerable: true,
        configurable: true
    });
    TinyYolov2Trainable.prototype.backward = function (img, groundTruth, inputSize, options) {
        if (options === void 0) { options = {}; }
        return tslib_1.__awaiter(this, void 0, void 0, function () {
            var _this = this;
            var _a, minBoxSize, reportLosses, reshapedImgDims, filteredGroundTruthBoxes, netInput, loss;
            return tslib_1.__generator(this, function (_b) {
                switch (_b.label) {
                    case 0:
                        _a = getDefaultBackwardOptions_1.getDefaultBackwardOptions(options), minBoxSize = _a.minBoxSize, reportLosses = _a.reportLosses;
                        reshapedImgDims = tfjs_image_recognition_base_1.computeReshapedDimensions(tfjs_image_recognition_base_1.getMediaDimensions(img), inputSize);
                        filteredGroundTruthBoxes = this.filterGroundTruthBoxes(groundTruth, reshapedImgDims, minBoxSize);
                        if (!filteredGroundTruthBoxes.length) {
                            return [2 /*return*/, null];
                        }
                        return [4 /*yield*/, tfjs_image_recognition_base_1.toNetInput(tfjs_image_recognition_base_1.imageToSquare(img, inputSize))];
                    case 1:
                        netInput = _b.sent();
                        loss = this.optimizer.minimize(function () {
                            var _a = _this.computeLoss(_this.forwardInput(netInput, inputSize), filteredGroundTruthBoxes, reshapedImgDims), noObjectLoss = _a.noObjectLoss, objectLoss = _a.objectLoss, coordLoss = _a.coordLoss, classLoss = _a.classLoss, totalLoss = _a.totalLoss;
                            if (reportLosses) {
                                var losses = {
                                    totalLoss: totalLoss.dataSync()[0],
                                    noObjectLoss: noObjectLoss.dataSync()[0],
                                    objectLoss: objectLoss.dataSync()[0],
                                    coordLoss: coordLoss.dataSync()[0],
                                    classLoss: classLoss.dataSync()[0]
                                };
                                var report = {
                                    losses: losses,
                                    numBoxes: filteredGroundTruthBoxes.length,
                                    inputSize: inputSize
                                };
                                reportLosses(report);
                            }
                            return totalLoss;
                        }, true);
                        return [2 /*return*/, loss];
                }
            });
        });
    };
    TinyYolov2Trainable.prototype.computeLoss = function (outputTensor, groundTruth, reshapedImgDims) {
        var config = config_1.validateTrainConfig(this.config);
        var inputSize = Math.max(reshapedImgDims.width, reshapedImgDims.height);
        if (!inputSize) {
            throw new Error("computeLoss - invalid inputSize: " + inputSize);
        }
        var predictedBoxes = this.extractBoxes(outputTensor, reshapedImgDims);
        return tf.tidy(function () {
            var lossFunction = new TinyYolov2LossFunction_1.TinyYolov2LossFunction(outputTensor, groundTruth, predictedBoxes, reshapedImgDims, config);
            return lossFunction.computeLoss();
        });
    };
    TinyYolov2Trainable.prototype.filterGroundTruthBoxes = function (groundTruth, imgDims, minBoxSize) {
        var imgHeight = imgDims.height, imgWidth = imgDims.width;
        return groundTruth.filter(function (_a) {
            var x = _a.x, y = _a.y, width = _a.width, height = _a.height;
            var box = (new tfjs_image_recognition_base_1.Rect(x, y, width, height))
                .rescale({ height: imgHeight, width: imgWidth });
            var isTooTiny = box.width < minBoxSize || box.height < minBoxSize;
            return !isTooTiny;
        });
    };
    TinyYolov2Trainable.prototype.load = function (weightsOrUrl) {
        return tslib_1.__awaiter(this, void 0, void 0, function () {
            return tslib_1.__generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, _super.prototype.load.call(this, weightsOrUrl)];
                    case 1:
                        _a.sent();
                        this.variable();
                        return [2 /*return*/];
                }
            });
        });
    };
    return TinyYolov2Trainable;
}(TinyYolov2_1.TinyYolov2));
exports.TinyYolov2Trainable = TinyYolov2Trainable;
//# sourceMappingURL=TinyYolov2Trainable.js.map