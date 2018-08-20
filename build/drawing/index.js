"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var utils_1 = require("../utils");
function getDefaultDrawOptions(options) {
    if (options === void 0) { options = {}; }
    return Object.assign({}, {
        boxColor: 'blue',
        textColor: 'red',
        lineWidth: 2,
        fontSize: 20,
        fontStyle: 'Georgia',
        withScore: true,
        withClassName: true
    }, options);
}
exports.getDefaultDrawOptions = getDefaultDrawOptions;
function drawBox(ctx, x, y, w, h, options) {
    var drawOptions = Object.assign(getDefaultDrawOptions(), (options || {}));
    ctx.strokeStyle = drawOptions.boxColor;
    ctx.lineWidth = drawOptions.lineWidth;
    ctx.strokeRect(x, y, w, h);
}
exports.drawBox = drawBox;
function drawText(ctx, x, y, text, options) {
    var drawOptions = Object.assign(getDefaultDrawOptions(), (options || {}));
    var padText = 2 + drawOptions.lineWidth;
    ctx.fillStyle = drawOptions.textColor;
    ctx.font = drawOptions.fontSize + "px " + drawOptions.fontStyle;
    ctx.fillText(text, x + padText, y + padText + (drawOptions.fontSize * 0.6));
}
exports.drawText = drawText;
function drawDetection(canvasArg, detection, options) {
    var canvas = utils_1.resolveInput(canvasArg);
    if (!(canvas instanceof HTMLCanvasElement)) {
        throw new Error('drawBox - expected canvas to be of type: HTMLCanvasElement');
    }
    var detectionArray = Array.isArray(detection)
        ? detection
        : [detection];
    detectionArray.forEach(function (det) {
        var _a = det.getBox(), x = _a.x, y = _a.y, width = _a.width, height = _a.height;
        var drawOptions = getDefaultDrawOptions(options);
        var ctx = utils_1.getContext2dOrThrow(canvas);
        drawBox(ctx, x, y, width, height, drawOptions);
        if (drawOptions.withScore) {
            drawText(ctx, x, y, det.className + " (" + utils_1.round(det.score) + ")", drawOptions);
        }
    });
}
exports.drawDetection = drawDetection;
//# sourceMappingURL=index.js.map