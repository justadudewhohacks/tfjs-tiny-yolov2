"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var Rect_1 = require("./Rect");
var ObjectDetection = /** @class */ (function () {
    function ObjectDetection(score, className, relativeBox, imageDims) {
        var width = imageDims.width, height = imageDims.height;
        this._imageWidth = width;
        this._imageHeight = height;
        this._score = score;
        this._className = className;
        this._box = new Rect_1.Rect(relativeBox.x * width, relativeBox.y * height, relativeBox.width * width, relativeBox.height * height);
    }
    Object.defineProperty(ObjectDetection.prototype, "score", {
        get: function () {
            return this._score;
        },
        enumerable: true,
        configurable: true
    });
    Object.defineProperty(ObjectDetection.prototype, "className", {
        get: function () {
            return this._className;
        },
        enumerable: true,
        configurable: true
    });
    Object.defineProperty(ObjectDetection.prototype, "box", {
        get: function () {
            return this._box;
        },
        enumerable: true,
        configurable: true
    });
    Object.defineProperty(ObjectDetection.prototype, "imageWidth", {
        get: function () {
            return this._imageWidth;
        },
        enumerable: true,
        configurable: true
    });
    Object.defineProperty(ObjectDetection.prototype, "imageHeight", {
        get: function () {
            return this._imageHeight;
        },
        enumerable: true,
        configurable: true
    });
    Object.defineProperty(ObjectDetection.prototype, "relativeBox", {
        get: function () {
            return new Rect_1.Rect(this._box.x / this._imageWidth, this._box.y / this._imageHeight, this._box.width / this._imageWidth, this._box.height / this._imageHeight);
        },
        enumerable: true,
        configurable: true
    });
    ObjectDetection.prototype.getScore = function () {
        return this.score;
    };
    ObjectDetection.prototype.getBox = function () {
        return this.box;
    };
    ObjectDetection.prototype.getImageWidth = function () {
        return this.imageWidth;
    };
    ObjectDetection.prototype.getImageHeight = function () {
        return this.imageHeight;
    };
    ObjectDetection.prototype.getRelativeBox = function () {
        return this.relativeBox;
    };
    ObjectDetection.prototype.forSize = function (width, height) {
        return new ObjectDetection(this.score, this.className, this.getRelativeBox(), { width: width, height: height });
    };
    return ObjectDetection;
}());
exports.ObjectDetection = ObjectDetection;
//# sourceMappingURL=ObjectDetection.js.map