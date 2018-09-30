"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var types_1 = require("./types");
var const_1 = require("./const");
function getDefaultForwardParams(params) {
    return Object.assign({}, {
        inputSize: types_1.SizeType.MD,
        scoreThreshold: 0.5
    }, params);
}
exports.getDefaultForwardParams = getDefaultForwardParams;
function getDefaultBackwardOptions(options) {
    return Object.assign({}, {
        minBoxSize: const_1.CELL_SIZE
    }, options);
}
exports.getDefaultBackwardOptions = getDefaultBackwardOptions;
//# sourceMappingURL=getDefaults.js.map