/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
package visrec.djl.ml.detection;

import ai.djl.Application;
import ai.djl.MalformedModelException;
import ai.djl.modality.cv.DetectedObjects;
import ai.djl.modality.cv.util.BufferedImageUtils;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import javax.visrec.ml.ClassificationException;
import javax.visrec.util.BoundingBox;
import org.testng.annotations.Test;

public class ObjectDetectorTest {

    @Test
    public void testObjectDetection()
            throws ClassificationException, IOException, ModelNotFoundException,
                    MalformedModelException {
        Criteria<BufferedImage, DetectedObjects> criteria =
                Criteria.builder()
                        .setTypes(BufferedImage.class, DetectedObjects.class)
                        .optApplication(Application.CV.OBJECT_DETECTION)
                        .optArgument("threshold", 0.01)
                        .build();
        try (ZooModel<BufferedImage, DetectedObjects> model = ModelZoo.loadModel(criteria)) {
            SimpleObjectDetector objectDetector = new SimpleObjectDetector(model);

            BufferedImage input =
                    BufferedImageUtils.fromUrl(
                            "https://djl-ai.s3.amazonaws.com/resources/images/dog_bike_car.jpg");
            Map<String, List<BoundingBox>> result = objectDetector.detectObject(input);

            for (List<BoundingBox> boundingBoxes : result.values()) {
                for (BoundingBox boundingBox : boundingBoxes) {
                    System.out.println(boundingBox.toString());
                }
            }
        }
    }
}
