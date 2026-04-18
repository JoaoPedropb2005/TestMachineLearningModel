/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package com.mycompany.testdensenet121;

/**
 *
 * @author jppb2
 */
import ai.onnxruntime.*;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import java.util.Collections;

public class DensenetDetector {
    private OrtEnvironment env;
    private OrtSession session;

    public DensenetDetector(String modelPath) throws OrtException {
        this.env = OrtEnvironment.getEnvironment();
        this.session = env.createSession(modelPath, new OrtSession.SessionOptions());
    }

    public String analisarImagem(String imagePath) throws Exception {
        
        Mat img = Imgcodecs.imread(imagePath);

        Imgproc.resize(img, img, new Size(224, 224));
        Imgproc.cvtColor(img, img, Imgproc.COLOR_BGR2RGB);

        float[] floatPixels = new float[1 * 3 * 224 * 224];
        int index = 0;
        for (int c = 0; c < 3; c++) {
            for (int h = 0; h < 224; h++) {
                for (int w = 0; w < 224; w++) {
                    double[] pixel = img.get(h, w);
                    floatPixels[index++] = (float) (pixel[c] / 255.0);
                }
            }
        }

        OnnxTensor inputTensor = OnnxTensor.createTensor(env, 
            java.nio.FloatBuffer.wrap(floatPixels), new long[]{1, 3, 224, 224});

        try (OrtSession.Result results = session.run(Collections.singletonMap("input_1", inputTensor))) {
            float[][] output = (float[][]) results.get(0).getValue();
            return resultadoAnalise(output[0]);
        }
    }

    private String resultadoAnalise(float[] prob) {
        if (prob[1] > prob[0]) {
            return String.format("Pneumonia detectada (%.2f%%)", prob[1] * 100);
        } else {
            return String.format("Exame saudável (%.2f%%)", prob[0] * 100);
        }
    }
}
