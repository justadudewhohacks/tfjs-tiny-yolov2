const dataFiles = [
].map(pattern => ({
  pattern,
  watched: false,
  included: false,
  served: true,
  nocache: false
}))

module.exports = function(config) {
  config.set({
    frameworks: ['jasmine', 'karma-typescript'],
    files: [
      'src/**/*.ts',
      'test/**/*.ts'
    ].concat(dataFiles),
    preprocessors: {
      '**/*.ts': ['karma-typescript']
    },
    karmaTypescriptConfig: {
      tsconfig: 'tsconfig.test.json',
      bundlerOptions: {
        transforms: [
          require("karma-typescript-es6-transform")()
        ]
      }
    },
    browsers: ['Chrome'],
    browserNoActivityTimeout: 60000,
    client: {
      jasmine: {
        timeoutInterval: 30000
      }
    }
  })
}
