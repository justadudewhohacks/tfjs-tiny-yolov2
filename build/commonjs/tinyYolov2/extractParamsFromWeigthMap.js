"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tfjs_image_recognition_base_1 = require("tfjs-image-recognition-base");
var extractSeparableConvParamsFactory_1 = require("../common/extractSeparableConvParamsFactory");
function extractorsFactory(weightMap, paramMappings) {
    var extractWeightEntry = tfjs_image_recognition_base_1.extractWeightEntryFactory(weightMap, paramMappings);
    function extractBatchNormParams(prefix) {
        var sub = extractWeightEntry(prefix + "/sub", 1);
        var truediv = extractWeightEntry(prefix + "/truediv", 1);
        return { sub: sub, truediv: truediv };
    }
    function extractConvParams(prefix) {
        var filters = extractWeightEntry(prefix + "/filters", 4);
        var bias = extractWeightEntry(prefix + "/bias", 1);
        return { filters: filters, bias: bias };
    }
    function extractConvWithBatchNormParams(prefix) {
        var conv = extractConvParams(prefix + "/conv");
        var bn = extractBatchNormParams(prefix + "/bn");
        return { conv: conv, bn: bn };
    }
    var extractSeparableConvParams = extractSeparableConvParamsFactory_1.loadSeparableConvParamsFactory(extractWeightEntry);
    return {
        extractConvParams: extractConvParams,
        extractConvWithBatchNormParams: extractConvWithBatchNormParams,
        extractSeparableConvParams: extractSeparableConvParams
    };
}
function extractParamsFromWeigthMap(weightMap, config) {
    var paramMappings = [];
    var _a = extractorsFactory(weightMap, paramMappings), extractConvParams = _a.extractConvParams, extractConvWithBatchNormParams = _a.extractConvWithBatchNormParams, extractSeparableConvParams = _a.extractSeparableConvParams;
    var params;
    if (config.withSeparableConvs) {
        var numFilters = (config.filterSizes && config.filterSizes.length || 9);
        params = {
            conv0: config.isFirstLayerConv2d ? extractConvParams('conv0') : extractSeparableConvParams('conv0'),
            conv1: extractSeparableConvParams('conv1'),
            conv2: extractSeparableConvParams('conv2'),
            conv3: extractSeparableConvParams('conv3'),
            conv4: extractSeparableConvParams('conv4'),
            conv5: extractSeparableConvParams('conv5'),
            conv6: numFilters > 7 ? extractSeparableConvParams('conv6') : undefined,
            conv7: numFilters > 8 ? extractSeparableConvParams('conv7') : undefined,
            conv8: extractConvParams('conv8')
        };
    }
    else {
        params = {
            conv0: extractConvWithBatchNormParams('conv0'),
            conv1: extractConvWithBatchNormParams('conv1'),
            conv2: extractConvWithBatchNormParams('conv2'),
            conv3: extractConvWithBatchNormParams('conv3'),
            conv4: extractConvWithBatchNormParams('conv4'),
            conv5: extractConvWithBatchNormParams('conv5'),
            conv6: extractConvWithBatchNormParams('conv6'),
            conv7: extractConvWithBatchNormParams('conv7'),
            conv8: extractConvParams('conv8')
        };
    }
    tfjs_image_recognition_base_1.disposeUnusedWeightTensors(weightMap, paramMappings);
    return { params: params, paramMappings: paramMappings };
}
exports.extractParamsFromWeigthMap = extractParamsFromWeigthMap;
//# sourceMappingURL=extractParamsFromWeigthMap.js.map