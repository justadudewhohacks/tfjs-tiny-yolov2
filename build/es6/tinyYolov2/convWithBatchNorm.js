import * as tf from '@tensorflow/tfjs-core';
import { SeparableConvParams } from '../common/types';
import { leaky } from './leaky';
export function convWithBatchNorm(x, params) {
    return tf.tidy(function () {
        var out = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]]);
        if (params instanceof SeparableConvParams) {
            out = tf.separableConv2d(out, params.depthwise_filter, params.pointwise_filter, [1, 1], 'valid');
            out = tf.add(out, params.bias);
        }
        else {
            out = tf.conv2d(out, params.conv.filters, [1, 1], 'valid');
            out = tf.sub(out, params.bn.sub);
            out = tf.mul(out, params.bn.truediv);
            out = tf.add(out, params.conv.bias);
        }
        return leaky(out);
    });
}
//# sourceMappingURL=convWithBatchNorm.js.map