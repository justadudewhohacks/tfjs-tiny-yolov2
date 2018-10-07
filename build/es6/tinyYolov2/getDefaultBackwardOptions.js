import { CELL_SIZE } from './const';
export function getDefaultBackwardOptions(options) {
    return Object.assign({}, {
        minBoxSize: CELL_SIZE
    }, options);
}
//# sourceMappingURL=getDefaultBackwardOptions.js.map