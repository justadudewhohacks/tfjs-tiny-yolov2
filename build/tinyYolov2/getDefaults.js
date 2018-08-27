import { SizeType } from './types';
import { CELL_SIZE } from './const';
export function getDefaultForwardParams(params) {
    return Object.assign({}, {
        inputSize: SizeType.MD,
        scoreThreshold: 0.5
    }, params);
}
export function getDefaultBackwardOptions(options) {
    return Object.assign({}, {
        minBoxSize: CELL_SIZE
    }, options);
}
//# sourceMappingURL=getDefaults.js.map