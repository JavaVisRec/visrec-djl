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
package ai.djl.jsr381.classification;

import java.awt.image.BufferedImage;
import java.io.File;
import java.net.URL;
import java.util.Map;
import javax.visrec.ml.ClassificationException;
import javax.visrec.ml.ClassifierCreationException;
import javax.visrec.ml.classification.ImageClassifier;
import javax.visrec.ml.classification.NeuralNetImageClassifier;
import org.testng.annotations.Test;

public class ImageClassifierTest {

    @Test
    public void testImageClassifier() throws ClassifierCreationException, ClassificationException {
        URL url = BinaryClassifierTest.class.getResource("/0.png");
        File input = new File(url.getFile());

        ImageClassifier<BufferedImage> classifier =
                NeuralNetImageClassifier.builder()
                        .inputClass(BufferedImage.class)
                        .imageHeight(28)
                        .imageWidth(28)
                        .build();

        Map<String, Float> result = classifier.classify(input);
        for (Map.Entry<String, Float> entry : result.entrySet()) {
            System.out.println(entry.getKey() + ": " + entry.getValue());
        }
    }
}
