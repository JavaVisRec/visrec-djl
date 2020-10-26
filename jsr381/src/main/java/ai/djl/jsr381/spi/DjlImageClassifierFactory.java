package ai.djl.jsr381.spi;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.basicdataset.ImageFolder;
import ai.djl.basicmodelzoo.cv.classification.ResNetV1;
import ai.djl.jsr381.classification.SimpleImageClassifier;
import ai.djl.metric.Metrics;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.Image.Flag;
import ai.djl.modality.cv.transform.CenterCrop;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.ImageClassificationTranslator;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import javax.visrec.ml.ClassifierCreationException;
import javax.visrec.ml.classification.ImageClassifier;
import javax.visrec.ml.classification.NeuralNetImageClassifier;
import javax.visrec.spi.ImageClassifierFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DjlImageClassifierFactory implements ImageClassifierFactory<BufferedImage> {

    private static final Logger logger = LoggerFactory.getLogger(DjlImageClassifierFactory.class);

    @Override
    public Class<BufferedImage> getImageClass() {
        return BufferedImage.class;
    }

    @Override
    public ImageClassifier<BufferedImage> create(
            NeuralNetImageClassifier.BuildingBlock<BufferedImage> block)
            throws ClassifierCreationException {
        int width = block.getImageWidth();
        int height = block.getImageHeight();

        Model model = Model.newInstance("imageClassifier"); // TODO generate better model name
        ZooModel<Image, Classifications> zooModel;

        Path modelPath = block.getImportPath();
        if (modelPath != null) {
            // load pre-trained model from model zoo
            logger.info("Loading pre-trained model ...");

            try {
                model.load(modelPath);
                Flag flag = width < 50 ? Flag.GRAYSCALE : Flag.COLOR;
                Translator<Image, Classifications> translator =
                        ImageClassificationTranslator.builder()
                                .optFlag(flag)
                                .addTransform(new CenterCrop())
                                .addTransform(new Resize(width, height))
                                .addTransform(new ToTensor())
                                .optSynsetArtifactName("synset.txt")
                                .optApplySoftmax(true)
                                .build();
                zooModel = new ZooModel<>(model, translator);
            } catch (MalformedModelException | IOException e) {
                throw new ClassifierCreationException("Failed load model from model zoo.", e);
            }
        } else {
            try {
                zooModel = trainWithResnet(model, block);
            } catch (IOException | TranslateException e) {
                throw new ClassifierCreationException("Failed train model.", e);
            }
        }
        return new SimpleImageClassifier(zooModel, 5);
    }

    private ZooModel<Image, Classifications> trainWithResnet(
            Model model, NeuralNetImageClassifier.BuildingBlock<BufferedImage> block)
            throws IOException, TranslateException {
        int width = block.getImageWidth();
        int height = block.getImageHeight();
        int epochs = block.getMaxEpochs();
        int batch = 1;

        File trainingFile = block.getTrainingFile();
        if (trainingFile == null) {
            throw new IllegalArgumentException("TrainingFile is required.");
        }
        ImageFolder dataset =
                ImageFolder.builder()
                        .setSampling(batch, true)
                        .setRepositoryPath(trainingFile.toPath())
                        .addTransform(new CenterCrop(width, height))
                        .addTransform(new Resize(width, height))
                        .addTransform(new ToTensor())
                        .build();

        RandomAccessDataset[] set = dataset.randomSplit(9, 1);

        List<String> synset = dataset.getSynset();

        Block resNet18 =
                ResNetV1.builder()
                        .setImageShape(new Shape(3, width, height))
                        .setNumLayers(18)
                        .setOutSize(synset.size())
                        .build();
        model.setBlock(resNet18);

        Path exportDir = block.getExportPath();
        // setup training configuration
        DefaultTrainingConfig config =
                new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                        .addEvaluator(new Accuracy())
                        .addTrainingListeners(TrainingListener.Defaults.logging());

        try (Trainer trainer = model.newTrainer(config)) {
            trainer.setMetrics(new Metrics());
            // initialize trainer with proper input shape
            trainer.initialize(new Shape(1, 3, width, height));
            EasyTrain.fit(trainer, epochs, set[0], set[1]);
        }

        if (exportDir != null) {
            model.save(exportDir, model.getName());
        }

        Batch b = dataset.getData(model.getNDManager()).iterator().next();
        NDArray array = b.getData().singletonOrThrow();

        Translator<Image, Classifications> translator =
                ImageClassificationTranslator.builder()
                        .addTransform(new CenterCrop(width, height))
                        .addTransform(new Resize(width, height))
                        .addTransform(new ToTensor())
                        .optSynset(synset)
                        .optApplySoftmax(true)
                        .build();
        return new ZooModel<>(model, translator);
    }
}
