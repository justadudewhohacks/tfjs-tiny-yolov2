export var TinyYolov2SizeType;
(function (TinyYolov2SizeType) {
    TinyYolov2SizeType[TinyYolov2SizeType["XS"] = 224] = "XS";
    TinyYolov2SizeType[TinyYolov2SizeType["SM"] = 320] = "SM";
    TinyYolov2SizeType[TinyYolov2SizeType["MD"] = 416] = "MD";
    TinyYolov2SizeType[TinyYolov2SizeType["LG"] = 608] = "LG";
})(TinyYolov2SizeType || (TinyYolov2SizeType = {}));
var TinyYolov2Options = /** @class */ (function () {
    function TinyYolov2Options(_a) {
        var _b = _a === void 0 ? {} : _a, inputSize = _b.inputSize, scoreThreshold = _b.scoreThreshold;
        this._name = 'TinyYolov2Options';
        this._inputSize = inputSize || 416;
        this._scoreThreshold = scoreThreshold || 0.5;
        if (typeof this._inputSize !== 'number' || this._inputSize % 32 !== 0) {
            throw new Error(this._name + " - expected inputSize to be a number divisible by 32");
        }
        if (typeof this._scoreThreshold !== 'number' || this._scoreThreshold <= 0 || this._scoreThreshold >= 1) {
            throw new Error(this._name + " - expected scoreThreshold to be a number between 0 and 1");
        }
    }
    Object.defineProperty(TinyYolov2Options.prototype, "inputSize", {
        get: function () { return this._inputSize; },
        enumerable: true,
        configurable: true
    });
    Object.defineProperty(TinyYolov2Options.prototype, "scoreThreshold", {
        get: function () { return this._scoreThreshold; },
        enumerable: true,
        configurable: true
    });
    return TinyYolov2Options;
}());
export { TinyYolov2Options };
//# sourceMappingURL=TinyYolov2Options.js.map