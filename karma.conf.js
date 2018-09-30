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
      tsconfig: 'tsconfig.test.json'
    },
    browsers: process.env.KARMA_BROWSERS
      ? process.env.KARMA_BROWSERS.split(',')
      : ['Chrome'],
    browserNoActivityTimeout: 120000,
    captureTimeout: 60000,
    client: {
      jasmine: {
        timeoutInterval: 60000
      }
    },
    customLaunchers: {
      ChromeNoSandbox: {
        base: 'Chrome',
        flags: ['--no-sandbox']
      }
    }
  })
}
