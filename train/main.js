const tf = require('@tensorflow/tfjs-node-gpu');

const data = require('./data');
const model = require('./model');

async function run(epochs, batchSize, modelSavePath) {
  data.loadData();

  const {images: trainImages, labels: trainLabels} = data.getTrainData();
  console.log("Training Images (Shape): " + trainImages.shape);
  console.log("Training Labels (Shape): " + trainLabels.shape);

  model.summary();

  const validationSplit = 0.15;
  const  onTrainBegin = logs => {console.log("onTrainBegin");}
  await model.fit(trainImages, trainLabels, {
    epochs,
    batchSize,
    validationSplit,
    callbacks: [tf.callbacks.earlyStopping({monitor:'val_loss', mode:'min', verbose:1, patience:25})]
  });

  const {images: testImages, labels: testLabels} = data.getTestData();
  const evalOutput = model.evaluate(testImages, testLabels);

  console.log(
      `\nEvaluation result:\n` +
      `  Loss = ${evalOutput[0].dataSync()[0].toFixed(3)}; `+
      `Accuracy = ${evalOutput[1].dataSync()[0].toFixed(3)}`);

  if (modelSavePath != null) {
    await model.save(`file://${modelSavePath}`);
    console.log(`Saved model to path: ${modelSavePath}`);
  }
}

run(600, 32, '../web/static/model');
