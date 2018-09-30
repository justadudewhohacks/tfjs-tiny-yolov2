function createLossReporter(trainSizes) {

  const totalLosses = {}
  const noObjectLosses = {}
  const objectLosses = {}
  const classLosses = {}
  const coordLosses = {}
  const counters = {}

  function resetLosses() {
    trainSizes.forEach(size => {
      totalLosses[size] = 0
      noObjectLosses[size] = 0
      objectLosses[size] = 0
      classLosses[size] = 0
      coordLosses[size] = 0
      counters[size] = 0
    })
  }

  function reportLosses({ losses, numBoxes, inputSize }) {
    totalLosses[inputSize] += losses.totalLoss
    noObjectLosses[inputSize] += losses.noObjectLoss
    objectLosses[inputSize] += losses.objectLoss
    classLosses[inputSize] += losses.classLoss
    coordLosses[inputSize] += losses.coordLoss
    counters[inputSize] += 1
  }

  function summary() {
    const avgLosses = {}

    trainSizes.forEach(size => {
      avgLosses[size] = {
        count: counters[size],
        totalLoss: totalLosses[size] / counters[size],
        noObjectLoss: noObjectLosses[size] / counters[size],
        objectLoss: objectLosses[size] / counters[size],
        classLoss: classLosses[size] / counters[size],
        coordLoss: coordLosses[size] / counters[size]
      }
    })

    return avgLosses
  }

  resetLosses()

  return {
    resetLosses,
    reportLosses,
    summary
  }
}