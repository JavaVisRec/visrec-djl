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

import java.io.File;
import javax.visrec.ml.ClassificationException;
import javax.visrec.ml.ClassifierCreationException;
import javax.visrec.ml.classification.BinaryClassifier;
import javax.visrec.ml.classification.NeuralNetBinaryClassifier;

public class BinaryClassifierExample {

    public static void main(String[] args)
            throws ClassificationException, ClassifierCreationException {
        File trainingFile = new File("../jsr381/src/test/resources/spam.csv");
        BinaryClassifier<float[]> spamClassifier =
                NeuralNetBinaryClassifier.builder()
                        .inputClass(float[].class)
                        .inputsNum(57)
                        .hiddenLayers(5)
                        .maxEpochs(2)
                        .trainingFile(trainingFile)
                        .build();

        // create test email feature
        float[] emailFeatures = new float[57];
        emailFeatures[56] = 1;

        Float result = spamClassifier.classify(emailFeatures);
        System.out.println(result);
    }
}
