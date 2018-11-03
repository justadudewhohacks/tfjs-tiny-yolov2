"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tf = require("@tensorflow/tfjs-core");
var tfjs_image_recognition_base_1 = require("tfjs-image-recognition-base");
var const_1 = require("./const");
var TinyYolov2LossFunction = /** @class */ (function () {
    function TinyYolov2LossFunction(outputTensor, groundTruth, predictedBoxes, reshapedImgDims, config) {
        this._config = config;
        this._reshapedImgDims = new tfjs_image_recognition_base_1.Dimensions(reshapedImgDims.width, reshapedImgDims.height);
        this._outputTensor = outputTensor;
        this._predictedBoxes = predictedBoxes;
        this.validateGroundTruthBoxes(groundTruth);
        this._groundTruth = this.assignGroundTruthToAnchors(groundTruth);
        var groundTruthMask = this.createGroundTruthMask();
        var _a = this.createCoordAndScoreMasks(), coordBoxOffsetMask = _a.coordBoxOffsetMask, coordBoxSizeMask = _a.coordBoxSizeMask, scoreMask = _a.scoreMask;
        this.noObjectLossMask = tf.tidy(function () { return tf.mul(scoreMask, tf.sub(tf.scalar(1), groundTruthMask)); });
        this.objectLossMask = tf.tidy(function () { return tf.mul(scoreMask, groundTruthMask); });
        this.coordBoxOffsetMask = tf.tidy(function () { return tf.mul(coordBoxOffsetMask, groundTruthMask); });
        this.coordBoxSizeMask = tf.tidy(function () { return tf.mul(coordBoxSizeMask, groundTruthMask); });
        var classScoresMask = tf.tidy(function () { return tf.sub(tf.scalar(1), coordBoxOffsetMask.add(coordBoxSizeMask).add(scoreMask)); });
        this.groundTruthClassScoresMask = tf.tidy(function () { return tf.mul(classScoresMask, groundTruthMask); });
    }
    Object.defineProperty(TinyYolov2LossFunction.prototype, "config", {
        get: function () {
            return this._config;
        },
        enumerable: true,
        configurable: true
    });
    Object.defineProperty(TinyYolov2LossFunction.prototype, "reshapedImgDims", {
        get: function () {
            return this._reshapedImgDims;
        },
        enumerable: true,
        configurable: true
    });
    Object.defineProperty(TinyYolov2LossFunction.prototype, "outputTensor", {
        get: function () {
            return this._outputTensor;
        },
        enumerable: true,
        configurable: true
    });
    Object.defineProperty(TinyYolov2LossFunction.prototype, "groundTruth", {
        get: function () {
            return this._groundTruth;
        },
        enumerable: true,
        configurable: true
    });
    Object.defineProperty(TinyYolov2LossFunction.prototype, "predictedBoxes", {
        get: function () {
            return this._predictedBoxes;
        },
        enumerable: true,
        configurable: true
    });
    Object.defineProperty(TinyYolov2LossFunction.prototype, "inputSize", {
        get: function () {
            return Math.max(this.reshapedImgDims.width, this.reshapedImgDims.height);
        },
        enumerable: true,
        configurable: true
    });
    Object.defineProperty(TinyYolov2LossFunction.prototype, "withClassScores", {
        get: function () {
            return this._config.withClassScores || this._config.classes.length > 1;
        },
        enumerable: true,
        configurable: true
    });
    Object.defineProperty(TinyYolov2LossFunction.prototype, "boxEncodingSize", {
        get: function () {
            return 5 + (this.withClassScores ? this._config.classes.length : 0);
        },
        enumerable: true,
        configurable: true
    });
    Object.defineProperty(TinyYolov2LossFunction.prototype, "anchors", {
        get: function () {
            return this._config.anchors;
        },
        enumerable: true,
        configurable: true
    });
    Object.defineProperty(TinyYolov2LossFunction.prototype, "numBoxes", {
        get: function () {
            return this.anchors.length;
        },
        enumerable: true,
        configurable: true
    });
    Object.defineProperty(TinyYolov2LossFunction.prototype, "numCells", {
        get: function () {
            return this.inputSize / const_1.CELL_SIZE;
        },
        enumerable: true,
        configurable: true
    });
    Object.defineProperty(TinyYolov2LossFunction.prototype, "gridCellEncodingSize", {
        get: function () {
            return this.boxEncodingSize * this.numBoxes;
        },
        enumerable: true,
        configurable: true
    });
    TinyYolov2LossFunction.prototype.toOutputTensorShape = function (tensor) {
        var _this = this;
        return tf.tidy(function () { return tensor.reshape([1, _this.numCells, _this.numCells, _this.gridCellEncodingSize]); });
    };
    TinyYolov2LossFunction.prototype.computeLoss = function () {
        var _this = this;
        return tf.tidy(function () {
            var noObjectLoss = _this.computeNoObjectLoss();
            var objectLoss = _this.computeObjectLoss();
            var coordLoss = _this.computeCoordLoss();
            var classLoss = _this.withClassScores
                ? _this.computeClassLoss()
                : tf.scalar(0);
            var totalLoss = tf.tidy(function () { return noObjectLoss.add(objectLoss).add(coordLoss).add(classLoss); });
            return {
                noObjectLoss: noObjectLoss,
                objectLoss: objectLoss,
                coordLoss: coordLoss,
                classLoss: classLoss,
                totalLoss: totalLoss
            };
        });
    };
    TinyYolov2LossFunction.prototype.computeNoObjectLoss = function () {
        var _this = this;
        return tf.tidy(function () {
            return _this.computeLossTerm(_this.config.noObjectScale, _this.toOutputTensorShape(_this.noObjectLossMask), tf.sigmoid(_this.outputTensor));
        });
    };
    TinyYolov2LossFunction.prototype.computeObjectLoss = function () {
        var _this = this;
        return tf.tidy(function () {
            return _this.computeLossTerm(_this.config.objectScale, _this.toOutputTensorShape(_this.objectLossMask), tf.sub(_this.toOutputTensorShape(_this.computeIous()), tf.sigmoid(_this.outputTensor)));
        });
    };
    TinyYolov2LossFunction.prototype.computeClassLoss = function () {
        var _this = this;
        return tf.tidy(function () {
            var classLossTensor = tf.tidy(function () {
                var predClassScores = tf.mul(tf.softmax(_this.outputTensor.reshape([_this.numCells, _this.numCells, _this.numBoxes, _this.boxEncodingSize]), 3), _this.groundTruthClassScoresMask);
                var gtClassScores = _this.createOneHotClassScoreMask();
                return tf.sub(gtClassScores, predClassScores);
            });
            return _this.computeLossTerm(_this.config.classScale, tf.scalar(1), classLossTensor);
        });
    };
    TinyYolov2LossFunction.prototype.computeCoordLoss = function () {
        var _this = this;
        return tf.tidy(function () {
            return _this.computeLossTerm(_this.config.coordScale, tf.scalar(1), tf.add(_this.computeCoordBoxOffsetError(), _this.computeCoordBoxSizeError()));
        });
    };
    TinyYolov2LossFunction.prototype.computeCoordBoxOffsetError = function () {
        var _this = this;
        return tf.tidy(function () {
            var mask = _this.toOutputTensorShape(_this.coordBoxOffsetMask);
            var gtBoxOffsets = tf.mul(mask, _this.toOutputTensorShape(_this.computeCoordBoxOffsets()));
            var predBoxOffsets = tf.mul(mask, tf.sigmoid(_this.outputTensor));
            return tf.sub(gtBoxOffsets, predBoxOffsets);
        });
    };
    TinyYolov2LossFunction.prototype.computeCoordBoxSizeError = function () {
        var _this = this;
        return tf.tidy(function () {
            var mask = _this.toOutputTensorShape(_this.coordBoxSizeMask);
            var gtBoxSizes = tf.mul(mask, _this.toOutputTensorShape(_this.computeCoordBoxSizes()));
            var predBoxSizes = tf.mul(mask, _this.outputTensor);
            return tf.sub(gtBoxSizes, predBoxSizes);
        });
    };
    TinyYolov2LossFunction.prototype.computeLossTerm = function (scale, mask, lossTensor) {
        var _this = this;
        return tf.tidy(function () { return tf.mul(tf.scalar(scale), _this.squaredSumOverMask(mask, lossTensor)); });
    };
    TinyYolov2LossFunction.prototype.squaredSumOverMask = function (mask, lossTensor) {
        return tf.tidy(function () { return tf.sum(tf.square(tf.mul(mask, lossTensor))); });
    };
    TinyYolov2LossFunction.prototype.validateGroundTruthBoxes = function (groundTruth) {
        var _this = this;
        groundTruth.forEach(function (_a) {
            var x = _a.x, y = _a.y, width = _a.width, height = _a.height, label = _a.label;
            if (typeof label !== 'number' || label < 0 || label > (_this.config.classes.length - 1)) {
                throw new Error("invalid ground truth data, expected label to be a number in [0, " + (_this.config.classes.length - 1) + "]");
            }
            if (x < 0 || x > 1 || y < 0 || y > 1 || width < 0 || (x + width) > 1 || height < 0 || (y + height) > 1) {
                throw new Error("invalid ground truth data, box is out of image boundaries " + JSON.stringify({ x: x, y: y, width: width, height: height }));
            }
        });
    };
    TinyYolov2LossFunction.prototype.assignGroundTruthToAnchors = function (groundTruth) {
        var _this = this;
        var groundTruthBoxes = groundTruth
            .map(function (_a) {
            var x = _a.x, y = _a.y, width = _a.width, height = _a.height, label = _a.label;
            return ({
                box: new tfjs_image_recognition_base_1.Rect(x, y, width, height),
                label: label
            });
        });
        return groundTruthBoxes.map(function (_a) {
            var box = _a.box, label = _a.label;
            var _b = box.rescale(_this.reshapedImgDims), left = _b.left, top = _b.top, bottom = _b.bottom, right = _b.right, x = _b.x, y = _b.y, width = _b.width, height = _b.height;
            var ctX = left + (width / 2);
            var ctY = top + (height / 2);
            var col = Math.floor((ctX / _this.inputSize) * _this.numCells);
            var row = Math.floor((ctY / _this.inputSize) * _this.numCells);
            var anchorsByIou = _this.anchors.map(function (anchor, idx) { return ({
                idx: idx,
                iou: tfjs_image_recognition_base_1.iou(new tfjs_image_recognition_base_1.BoundingBox(0, 0, anchor.x * const_1.CELL_SIZE, anchor.y * const_1.CELL_SIZE), new tfjs_image_recognition_base_1.BoundingBox(0, 0, width, height))
            }); }).sort(function (a1, a2) { return a2.iou - a1.iou; });
            var anchor = anchorsByIou[0].idx;
            return { row: row, col: col, anchor: anchor, box: box, label: label };
        });
    };
    TinyYolov2LossFunction.prototype.createGroundTruthMask = function () {
        var _this = this;
        var mask = tf.zeros([this.numCells, this.numCells, this.numBoxes, this.boxEncodingSize]);
        var buf = mask.buffer();
        this.groundTruth.forEach(function (_a) {
            var row = _a.row, col = _a.col, anchor = _a.anchor;
            for (var i = 0; i < _this.boxEncodingSize; i++) {
                buf.set(1, row, col, anchor, i);
            }
        });
        return mask;
    };
    TinyYolov2LossFunction.prototype.createCoordAndScoreMasks = function () {
        var _this = this;
        return tf.tidy(function () {
            var coordBoxOffsetMask = tf.zeros([_this.numCells, _this.numCells, _this.numBoxes, _this.boxEncodingSize]);
            var coordBoxSizeMask = tf.zeros([_this.numCells, _this.numCells, _this.numBoxes, _this.boxEncodingSize]);
            var scoreMask = tf.zeros([_this.numCells, _this.numCells, _this.numBoxes, _this.boxEncodingSize]);
            var coordBoxOffsetBuf = coordBoxOffsetMask.buffer();
            var coordBoxSizeBuf = coordBoxSizeMask.buffer();
            var scoreBuf = scoreMask.buffer();
            for (var row = 0; row < _this.numCells; row++) {
                for (var col = 0; col < _this.numCells; col++) {
                    for (var anchor = 0; anchor < _this.numBoxes; anchor++) {
                        coordBoxOffsetBuf.set(1, row, col, anchor, 0);
                        coordBoxOffsetBuf.set(1, row, col, anchor, 1);
                        coordBoxSizeBuf.set(1, row, col, anchor, 2);
                        coordBoxSizeBuf.set(1, row, col, anchor, 3);
                        scoreBuf.set(1, row, col, anchor, 4);
                    }
                }
            }
            return { coordBoxOffsetMask: coordBoxOffsetMask, coordBoxSizeMask: coordBoxSizeMask, scoreMask: scoreMask };
        });
    };
    TinyYolov2LossFunction.prototype.createOneHotClassScoreMask = function () {
        var mask = tf.zeros([this.numCells, this.numCells, this.numBoxes, this.boxEncodingSize]);
        var buf = mask.buffer();
        var classValuesOffset = 5;
        this.groundTruth.forEach(function (_a) {
            var row = _a.row, col = _a.col, anchor = _a.anchor, label = _a.label;
            buf.set(1, row, col, anchor, classValuesOffset + label);
        });
        return mask;
    };
    TinyYolov2LossFunction.prototype.computeIous = function () {
        var _this = this;
        var isSameAnchor = function (p1) { return function (p2) {
            return p1.row === p2.row
                && p1.col === p2.col
                && p1.anchor === p2.anchor;
        }; };
        var ious = tf.zeros([this.numCells, this.numCells, this.gridCellEncodingSize]);
        var buf = ious.buffer();
        this.groundTruth.forEach(function (_a) {
            var row = _a.row, col = _a.col, anchor = _a.anchor, box = _a.box;
            var predBox = _this.predictedBoxes.find(isSameAnchor({ row: row, col: col, anchor: anchor }));
            if (!predBox) {
                throw new Error("no output box found for: row " + row + ", col " + col + ", anchor " + anchor);
            }
            var boxIou = tfjs_image_recognition_base_1.iou(box.rescale(_this.reshapedImgDims), predBox.box.rescale(_this.reshapedImgDims));
            var anchorOffset = _this.boxEncodingSize * anchor;
            var scoreValueOffset = 4;
            buf.set(boxIou, row, col, anchorOffset + scoreValueOffset);
        });
        return ious;
    };
    TinyYolov2LossFunction.prototype.computeCoordBoxOffsets = function () {
        var _this = this;
        var offsets = tf.zeros([this.numCells, this.numCells, this.numBoxes, this.boxEncodingSize]);
        var buf = offsets.buffer();
        this.groundTruth.forEach(function (_a) {
            var row = _a.row, col = _a.col, anchor = _a.anchor, box = _a.box;
            var _b = box.rescale(_this.reshapedImgDims), left = _b.left, top = _b.top, right = _b.right, bottom = _b.bottom;
            var centerX = (left + right) / 2;
            var centerY = (top + bottom) / 2;
            var dCenterX = centerX - (col * const_1.CELL_SIZE);
            var dCenterY = centerY - (row * const_1.CELL_SIZE);
            // inverseSigmoid(0) === -Infinity, inverseSigmoid(1) === Infinity
            //const dx = inverseSigmoid(Math.min(0.999, Math.max(0.001, dCenterX / CELL_SIZE)))
            //const dy = inverseSigmoid(Math.min(0.999, Math.max(0.001, dCenterY / CELL_SIZE)))
            var dx = dCenterX / const_1.CELL_SIZE;
            var dy = dCenterY / const_1.CELL_SIZE;
            buf.set(dx, row, col, anchor, 0);
            buf.set(dy, row, col, anchor, 1);
        });
        return offsets;
    };
    TinyYolov2LossFunction.prototype.computeCoordBoxSizes = function () {
        var _this = this;
        var sizes = tf.zeros([this.numCells, this.numCells, this.numBoxes, this.boxEncodingSize]);
        var buf = sizes.buffer();
        this.groundTruth.forEach(function (_a) {
            var row = _a.row, col = _a.col, anchor = _a.anchor, box = _a.box;
            var _b = box.rescale(_this.reshapedImgDims), width = _b.width, height = _b.height;
            var dw = Math.log(width / (_this.anchors[anchor].x * const_1.CELL_SIZE));
            var dh = Math.log(height / (_this.anchors[anchor].y * const_1.CELL_SIZE));
            buf.set(dw, row, col, anchor, 2);
            buf.set(dh, row, col, anchor, 3);
        });
        return sizes;
    };
    return TinyYolov2LossFunction;
}());
exports.TinyYolov2LossFunction = TinyYolov2LossFunction;
//# sourceMappingURL=TinyYolov2LossFunction.js.map