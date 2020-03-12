package ai.djl.jsr381.spi;

import ai.djl.MalformedModelException;
import ai.djl.basicmodelzoo.BasicModelZoo;
import ai.djl.jsr381.classification.SimpleImageClassifier;
import ai.djl.modality.Classifications;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
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
        File modelFile = block.getModelFile();
        if (modelFile == null) {
            // load pre-trained model from model zoo
            logger.info("Loading pre-trained model ...");

            try {
                Criteria<BufferedImage, Classifications> criteria =
                        Criteria.builder()
                                .setTypes(BufferedImage.class, Classifications.class)
                                .build();
                ZooModel<BufferedImage, Classifications> model =
                        BasicModelZoo.MLP.loadModel(criteria);
                return new SimpleImageClassifier(model, 5);
            } catch (ModelNotFoundException | MalformedModelException | IOException e) {
                throw new ClassifierCreationException("Failed load model from model zoo.", e);
            }
        }

        if (!modelFile.exists()) {
            // train the model first.
        }
        return null;
    }
}
