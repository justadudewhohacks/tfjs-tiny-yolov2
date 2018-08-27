var SeparableConvParams = /** @class */ (function () {
    function SeparableConvParams(depthwise_filter, pointwise_filter, bias) {
        this.depthwise_filter = depthwise_filter;
        this.pointwise_filter = pointwise_filter;
        this.bias = bias;
    }
    return SeparableConvParams;
}());
export { SeparableConvParams };
export var SizeType;
(function (SizeType) {
    SizeType["XS"] = "xs";
    SizeType["SM"] = "sm";
    SizeType["MD"] = "md";
    SizeType["LG"] = "lg";
})(SizeType || (SizeType = {}));
//# sourceMappingURL=types.js.map