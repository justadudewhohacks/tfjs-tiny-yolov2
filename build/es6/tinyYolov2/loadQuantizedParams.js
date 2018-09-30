import * as tslib_1 from "tslib";
import { disposeUnusedWeightTensors, extractWeightEntryFactory, loadWeightMap, } from 'tfjs-image-recognition-base';
import { loadSeparableConvParamsFactory } from '../common/extractSeparableConvParamsFactory';
function extractorsFactory(weightMap, paramMappings) {
    var extractWeightEntry = extractWeightEntryFactory(weightMap, paramMappings);
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
    var extractSeparableConvParams = loadSeparableConvParamsFactory(extractWeightEntry);
    return {
        extractConvParams: extractConvParams,
        extractConvWithBatchNormParams: extractConvWithBatchNormParams,
        extractSeparableConvParams: extractSeparableConvParams
    };
}
export function loadQuantizedParams(uri, withSeparableConvs, defaultModelName) {
    if (defaultModelName === void 0) { defaultModelName = ''; }
    return tslib_1.__awaiter(this, void 0, void 0, function () {
        var weightMap, paramMappings, _a, extractConvParams, extractConvWithBatchNormParams, extractSeparableConvParams, extractConvFn, params;
        return tslib_1.__generator(this, function (_b) {
            switch (_b.label) {
                case 0: return [4 /*yield*/, loadWeightMap(uri, defaultModelName)];
                case 1:
                    weightMap = _b.sent();
                    paramMappings = [];
                    _a = extractorsFactory(weightMap, paramMappings), extractConvParams = _a.extractConvParams, extractConvWithBatchNormParams = _a.extractConvWithBatchNormParams, extractSeparableConvParams = _a.extractSeparableConvParams;
                    extractConvFn = withSeparableConvs ? extractSeparableConvParams : extractConvWithBatchNormParams;
                    params = {
                        conv0: extractConvFn('conv0'),
                        conv1: extractConvFn('conv1'),
                        conv2: extractConvFn('conv2'),
                        conv3: extractConvFn('conv3'),
                        conv4: extractConvFn('conv4'),
                        conv5: extractConvFn('conv5'),
                        conv6: extractConvFn('conv6'),
                        conv7: extractConvFn('conv7'),
                        conv8: extractConvParams('conv8')
                    };
                    disposeUnusedWeightTensors(weightMap, paramMappings);
                    return [2 /*return*/, { params: params, paramMappings: paramMappings }];
            }
        });
    });
}
//# sourceMappingURL=loadQuantizedParams.js.map