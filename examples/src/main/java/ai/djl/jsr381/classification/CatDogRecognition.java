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

import ai.djl.util.ZipUtils;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;
import javax.visrec.ml.ClassifierCreationException;
import javax.visrec.ml.classification.ImageClassifier;
import javax.visrec.ml.classification.NeuralNetImageClassifier;

public class CatDogRecognition {

    public static void main(String[] args) throws IOException, ClassifierCreationException {
        File trainingFile = downloadTrainingData();
        Path modelDir = Paths.get("build/model");

        ImageClassifier<BufferedImage> classifier =
                NeuralNetImageClassifier.builder()
                        .inputClass(BufferedImage.class)
                        .imageHeight(128)
                        .imageWidth(128)
                        .trainingFile(trainingFile)
                        .exportModel(modelDir)
                        .maxEpochs(20)
                        .build();

        File input = new File(trainingFile, "cat/cat_1.png");
        Map<String, Float> result = classifier.classify(input);
        for (Map.Entry<String, Float> entry : result.entrySet()) {
            System.out.println(entry.getKey() + ": " + entry.getValue());
        }
    }

    private static File downloadTrainingData() throws IOException {
        String link =
                "https://github.com/JavaVisRec/jsr381-examples-datasets/raw/master/cats_and_dogs_training_data_png.zip";
        URL url = new URL(link);
        Path dir = Paths.get("datasets", "cats_and_dogs");
        if (!Files.exists(dir)) {
            Files.createDirectories(dir);
            try (InputStream is = url.openStream()) {
                ZipUtils.unzip(is, dir);
            }
        }
        return dir.resolve("training").toFile();
    }
}
