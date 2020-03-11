package ai.djl.jsr381.detection;

import ai.djl.inference.Predictor;
import ai.djl.modality.cv.DetectedObjects;
import ai.djl.modality.cv.Rectangle;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import javax.visrec.ml.ClassificationException;
import javax.visrec.ml.detection.ObjectDetector;
import javax.visrec.util.BoundingBox;

/** A simple object detector implemented with DJL. */
public class SimpleObjectDetector implements ObjectDetector<BufferedImage> {

    private ZooModel<BufferedImage, DetectedObjects> model;

    public SimpleObjectDetector(ZooModel<BufferedImage, DetectedObjects> model) {
        this.model = model;
    }

    @Override
    public Map<String, List<BoundingBox>> detectObject(BufferedImage image)
            throws ClassificationException {
        try (Predictor<BufferedImage, DetectedObjects> predictor = model.newPredictor()) {
            DetectedObjects detectedObjects = predictor.predict(image);
            Map<String, List<BoundingBox>> ret = new ConcurrentHashMap<>();

            int imageWidth = image.getWidth();
            int imageHeight = image.getHeight();

            List<DetectedObjects.DetectedObject> detections = detectedObjects.items();
            for (DetectedObjects.DetectedObject detection : detections) {
                String className = detection.getClassName();
                float probability = (float) detection.getProbability();
                Rectangle rect = detection.getBoundingBox().getBounds();

                int x = (int) (rect.getX() * imageWidth);
                int y = (int) (rect.getY() * imageHeight);
                float w = (float) (rect.getWidth() * imageWidth);
                float h = (float) (rect.getHeight() * imageHeight);

                ret.compute(
                        className,
                        (k, list) -> {
                            if (list == null) {
                                list = new ArrayList<>();
                            }
                            list.add(new BoundingBox(className, probability, x, y, w, h));
                            return list;
                        });
            }

            return ret;
        } catch (TranslateException e) {
            throw new ClassificationException("Failed to process output", e);
        }
    }
}
