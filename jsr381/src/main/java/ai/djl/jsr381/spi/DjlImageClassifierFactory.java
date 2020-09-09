package ai.djl.jsr381.spi;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.jsr381.classification.SimpleImageClassifier;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.Image.Flag;
import ai.djl.modality.cv.transform.CenterCrop;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.ImageClassificationTranslator;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.Pipeline;
import ai.djl.translate.Translator;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Path;
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
        Flag flag = width < 50 ? Flag.GRAYSCALE : Flag.COLOR;

        Path modelPath = block.getImportPath();
        if (modelPath != null) {
            // load pre-trained model from model zoo
            logger.info("Loading pre-trained model ...");

            try {
                Pipeline pipeline = new Pipeline();
                pipeline.add(new CenterCrop()).add(new Resize(width, height)).add(new ToTensor());
                Translator<Image, Classifications> translator =
                        ImageClassificationTranslator.builder()
                                .optFlag(flag)
                                .setPipeline(pipeline)
                                .optSynsetArtifactName("synset.txt")
                                .optApplySoftmax(true)
                                .build();

                Model model =
                        Model.newInstance("imageClassifier"); // TODO generate better model name
                model.load(modelPath);
                ZooModel<Image, Classifications> zooModel = new ZooModel<>(model, translator);
                return new SimpleImageClassifier(zooModel, 5);
            } catch (MalformedModelException | IOException e) {
                throw new ClassifierCreationException("Failed load model from model zoo.", e);
            }
        }

        return null;
    }
}
