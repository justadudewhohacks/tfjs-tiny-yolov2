"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tf = require("@tensorflow/tfjs-core");
var tfjs_image_recognition_base_1 = require("tfjs-image-recognition-base");
var common_1 = require("../common");
var extractSeparableConvParamsFactory_1 = require("../common/extractSeparableConvParamsFactory");
function extractorsFactory(extractWeights, paramMappings) {
    var extractConvParams = common_1.extractConvParamsFactory(extractWeights, paramMappings);
    function extractBatchNormParams(size, mappedPrefix) {
        var sub = tf.tensor1d(extractWeights(size));
        var truediv = tf.tensor1d(extractWeights(size));
        paramMappings.push({ paramPath: mappedPrefix + "/sub" }, { paramPath: mappedPrefix + "/truediv" });
        return { sub: sub, truediv: truediv };
    }
    function extractConvWithBatchNormParams(channelsIn, channelsOut, mappedPrefix) {
        var conv = extractConvParams(channelsIn, channelsOut, 3, mappedPrefix + "/conv");
        var bn = extractBatchNormParams(channelsOut, mappedPrefix + "/bn");
        return { conv: conv, bn: bn };
    }
    var extractSeparableConvParams = extractSeparableConvParamsFactory_1.extractSeparableConvParamsFactory(extractWeights, paramMappings);
    return {
        extractConvParams: extractConvParams,
        extractConvWithBatchNormParams: extractConvWithBatchNormParams,
        extractSeparableConvParams: extractSeparableConvParams
    };
}
function extractParams(weights, withSeparableConvs, boxEncodingSize, filterSizes) {
    var _a = tfjs_image_recognition_base_1.extractWeightsFactory(weights), extractWeights = _a.extractWeights, getRemainingWeights = _a.getRemainingWeights;
    var paramMappings = [];
    var _b = extractorsFactory(extractWeights, paramMappings), extractConvParams = _b.extractConvParams, extractConvWithBatchNormParams = _b.extractConvWithBatchNormParams, extractSeparableConvParams = _b.extractSeparableConvParams;
    var extractConvFn = withSeparableConvs ? extractSeparableConvParams : extractConvWithBatchNormParams;
    var s0 = filterSizes[0], s1 = filterSizes[1], s2 = filterSizes[2], s3 = filterSizes[3], s4 = filterSizes[4], s5 = filterSizes[5], s6 = filterSizes[6], s7 = filterSizes[7], s8 = filterSizes[8];
    var conv0 = extractConvFn(s0, s1, 'conv0');
    var conv1 = extractConvFn(s1, s2, 'conv1');
    var conv2 = extractConvFn(s2, s3, 'conv2');
    var conv3 = extractConvFn(s3, s4, 'conv3');
    var conv4 = extractConvFn(s4, s5, 'conv4');
    var conv5 = extractConvFn(s5, s6, 'conv5');
    var conv6 = extractConvFn(s6, s7, 'conv6');
    var conv7 = extractConvFn(s7, s8, 'conv7');
    var conv8 = extractConvParams(s8, 5 * boxEncodingSize, 1, 'conv8');
    if (getRemainingWeights().length !== 0) {
        throw new Error("weights remaing after extract: " + getRemainingWeights().length);
    }
    var params = { conv0: conv0, conv1: conv1, conv2: conv2, conv3: conv3, conv4: conv4, conv5: conv5, conv6: conv6, conv7: conv7, conv8: conv8 };
    return { params: params, paramMappings: paramMappings };
}
exports.extractParams = extractParams;
//# sourceMappingURL=extractParams.js.map