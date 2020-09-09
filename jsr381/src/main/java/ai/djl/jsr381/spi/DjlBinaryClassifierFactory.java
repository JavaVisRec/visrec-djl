package ai.djl.jsr381.spi;

import ai.djl.Model;
import ai.djl.jsr381.classification.SimpleBinaryClassifier;
import ai.djl.jsr381.dataset.CsvDataset;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Blocks;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.BatchNorm;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.evaluator.BinaryAccuracy;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.translate.Batchifier;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import java.io.IOException;
import javax.visrec.ml.ClassifierCreationException;
import javax.visrec.ml.classification.BinaryClassifier;
import javax.visrec.ml.classification.NeuralNetBinaryClassifier;
import javax.visrec.spi.BinaryClassifierFactory;

public class DjlBinaryClassifierFactory implements BinaryClassifierFactory<float[]> {

    @Override
    public Class<float[]> getTargetClass() {
        return float[].class;
    }

    @Override
    public BinaryClassifier<float[]> create(NeuralNetBinaryClassifier.BuildingBlock<float[]> block)
            throws ClassifierCreationException {
        int inputSize = block.getInputsNum();
        int[] hiddenLayers = block.getHiddenLayers();
        int epochs = block.getMaxEpochs();
        int batchSize = 32;

        SequentialBlock mlp = new SequentialBlock().add(Blocks.batchFlattenBlock(inputSize));
        for (int size : hiddenLayers) {
            mlp.add(Linear.builder().setUnits(size).build()).add(Activation::relu);
        }
        mlp.add(BatchNorm.builder().build())
                .add(Linear.builder().setUnits(1).build())
                .add(arrays -> new NDList(arrays.singletonOrThrow().flatten()));

        Model model = Model.newInstance("binaryClassifier"); // TODO generate better model name
        model.setBlock(mlp);

        RandomAccessDataset[] dataset;
        try {
            CsvDataset csv =
                    CsvDataset.builder()
                            .setCsvFile(block.getTrainingFile())
                            .setSampling(batchSize, true)
                            .build();
            dataset = csv.randomSplit(8, 2);
        } catch (IOException | TranslateException e) {
            throw new ClassifierCreationException("Failed to load dataset.", e);
        }

        // setup training configuration
        DefaultTrainingConfig config =
                new DefaultTrainingConfig(Loss.sigmoidBinaryCrossEntropyLoss())
                        .addTrainingListeners(TrainingListener.Defaults.logging())
                        .addEvaluator(new BinaryAccuracy());

        try (Trainer trainer = model.newTrainer(config)) {
            trainer.setMetrics(new Metrics());
            Shape inputShape = new Shape(1, inputSize);
            trainer.initialize(inputShape);

            for (int i = 0; i < epochs; i++) {
                for (Batch batch : trainer.iterateDataset(dataset[0])) {
                    EasyTrain.trainBatch(trainer, batch);
                    trainer.step();
                    batch.close();
                }

                for (Batch batch : trainer.iterateDataset(dataset[1])) {
                    EasyTrain.validateBatch(trainer, batch);
                    batch.close();
                }

                // reset training and validation evaluators at end of epoch
                trainer.notifyListeners(listener -> listener.onEpoch(trainer));
            }
        } catch (IOException | TranslateException e) {
            throw new ClassifierCreationException("Failed to process dataset.", e);
        }

        return new SimpleBinaryClassifier(new ZooModel<>(model, new BinaryClassifierTranslator()));
    }

    private static final class BinaryClassifierTranslator implements Translator<float[], Float> {

        @Override
        public NDList processInput(TranslatorContext ctx, float[] input) {
            NDManager manager = ctx.getNDManager();
            NDArray array = manager.create(input);
            return new NDList(array);
        }

        @Override
        public Float processOutput(TranslatorContext ctx, NDList list) {
            return list.singletonOrThrow().getFloat();
        }

        @Override
        public Batchifier getBatchifier() {
            return Batchifier.STACK;
        }
    }
}
