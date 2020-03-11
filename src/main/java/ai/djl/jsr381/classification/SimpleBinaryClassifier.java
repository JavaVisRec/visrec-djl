package ai.djl.jsr381.classification;

import ai.djl.inference.Predictor;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;
import javax.visrec.ml.ClassificationException;
import javax.visrec.ml.classification.BinaryClassifier;

/** Implementation of a {@link BinaryClassifier} with DJL. */
public class SimpleBinaryClassifier implements BinaryClassifier<float[]> {

    private ZooModel<float[], Float> model;

    public SimpleBinaryClassifier(ZooModel<float[], Float> model) {
        this.model = model;
    }

    @Override
    public Float classify(float[] input) throws ClassificationException {
        try (Predictor<float[], Float> predictor = model.newPredictor()) {
            return predictor.predict(input);
        } catch (TranslateException e) {
            throw new ClassificationException("Failed to process output", e);
        }
    }
}
