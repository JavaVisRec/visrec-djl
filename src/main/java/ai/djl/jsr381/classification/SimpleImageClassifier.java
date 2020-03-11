package ai.djl.jsr381.classification;

import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import javax.imageio.ImageIO;
import javax.visrec.ml.ClassificationException;
import javax.visrec.ml.classification.ImageClassifier;

/**
 * Implementation of abstract image classifier for BufferedImage-s using DJL.
 *
 * @author Frank Liu
 */
public class SimpleImageClassifier implements ImageClassifier<BufferedImage> {

    private ZooModel<BufferedImage, Classifications> model;
    private int topK;

    public SimpleImageClassifier(ZooModel<BufferedImage, Classifications> model, int topK) {
        this.model = model;
        this.topK = topK;
    }

    @Override
    public Map<String, Float> classify(File input) throws ClassificationException {
        try {
            return classify(ImageIO.read(input));
        } catch (IOException e) {
            throw new ClassificationException("Couldn't transform input into a BufferedImage", e);
        }
    }

    @Override
    public Map<String, Float> classify(InputStream input) throws ClassificationException {
        try {
            return classify(ImageIO.read(input));
        } catch (IOException e) {
            throw new ClassificationException("Couldn't transform input into a BufferedImage", e);
        }
    }

    @Override
    public Map<String, Float> classify(BufferedImage input) throws ClassificationException {
        try (Predictor<BufferedImage, Classifications> predictor = model.newPredictor()) {
            Classifications classifications = predictor.predict(input);
            List<Classifications.Classification> list = classifications.topK(topK);
            return list.stream()
                    .collect(
                            Collectors.toMap(
                                    Classifications.Classification::getClassName,
                                    x -> (float) x.getProbability()));
        } catch (TranslateException e) {
            throw new ClassificationException("Failed to process output", e);
        }
    }
}
