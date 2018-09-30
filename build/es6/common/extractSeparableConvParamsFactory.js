import * as tf from '@tensorflow/tfjs-core';
import { SeparableConvParams } from './types';
export function extractSeparableConvParamsFactory(extractWeights, paramMappings) {
    return function (channelsIn, channelsOut, mappedPrefix) {
        var depthwise_filter = tf.tensor4d(extractWeights(3 * 3 * channelsIn), [3, 3, channelsIn, 1]);
        var pointwise_filter = tf.tensor4d(extractWeights(channelsIn * channelsOut), [1, 1, channelsIn, channelsOut]);
        var bias = tf.tensor1d(extractWeights(channelsOut));
        paramMappings.push({ paramPath: mappedPrefix + "/depthwise_filter" }, { paramPath: mappedPrefix + "/pointwise_filter" }, { paramPath: mappedPrefix + "/bias" });
        return new SeparableConvParams(depthwise_filter, pointwise_filter, bias);
    };
}
export function loadSeparableConvParamsFactory(extractWeightEntry) {
    return function (prefix) {
        var depthwise_filter = extractWeightEntry(prefix + "/depthwise_filter", 4);
        var pointwise_filter = extractWeightEntry(prefix + "/pointwise_filter", 4);
        var bias = extractWeightEntry(prefix + "/bias", 1);
        return new SeparableConvParams(depthwise_filter, pointwise_filter, bias);
    };
}
//# sourceMappingURL=extractSeparableConvParamsFactory.js.map