/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.tensorflow.lite.examples.classification.tflite;

import android.app.Activity;
import android.graphics.RectF;
import android.util.Log;

import java.io.IOException;
import java.util.ArrayList;

/** This TensorFlowLite classifier works with the float MobileNet model. */
public class ClassifierFloatMobileNet extends Classifier {

  /** MobileNet requires additional normalization of the used input. */
  private static final float IMAGE_MEAN = 127.5f;
  private static final float IMAGE_STD = 127.5f;

  /**
   * An array to hold inference results, to be feed into Tensorflow Lite as outputs. This isn't part
   * of the super class, because we need a primitive array here.
   */
  private float[][][][] labelProbArray = null;
  private float[] poses_x = null;
  private float[] poses_y = null;
  private float[] tmp_max = null;
  private int num_classes = 17;
  private int output_stride = 16;

  /**
   * Initializes a {@code ClassifierFloatMobileNet}.
   *
   * @param activity
   */
  public ClassifierFloatMobileNet(Activity activity, Device device, int numThreads)
      throws IOException {
    super(activity, device, numThreads);
    //  labelProbArray = new float[1][getNumLabels()];
    labelProbArray = new float[1] [23] [17] [17];

    poses_x = new float[num_classes];
    poses_y = new float[num_classes];
    tmp_max = new float[num_classes];
    for(int i = 0; i < num_classes; i++){
      poses_x[0] = (float) 0.0;
      poses_y[0] = (float) 0.0;
      tmp_max[0] = (float) 0.0;
    }
  }

  @Override
  public int getImageSizeX() {
    //return 224;
    return 257;
  }

  @Override
  public int getImageSizeY() {
    //return 224;
    return 353;
  }

  @Override
  protected String getModelPath() {
    // you can download this file from
    // see build.gradle for where to obtain this file. It should be auto
    // downloaded into assets.
    //return "mobilenet_v1_1.0_224.tflite";
    return "multi_person_mobilenet_v1_075_float.tflite";
  }

  @Override
  protected String getLabelPath() {
    return "labels.txt";
  }

  @Override
  protected int getNumBytesPerChannel() {
    return 4; // Float.SIZE / Byte.SIZE;
  }

  @Override
  protected void addPixelValue(int pixelValue) {
    imgData.putFloat((((pixelValue >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
    imgData.putFloat((((pixelValue >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
    imgData.putFloat(((pixelValue & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
  }

  @Override
  protected float getProbability(int labelIndex) {
    return labelProbArray[0][labelIndex][0][0];
  }

  @Override
  protected void setProbability(int labelIndex, Number value) {
    Log.v("MG", "get float value at labelIndex = " + labelIndex);
    float result = value.floatValue();
    Log.v("MG", "result = " + result);
    labelProbArray[0][labelIndex][0][0] = result;
    Log.v("MG", "done");
  }

  @Override
  protected float getNormalizedProbability(int labelIndex) {
    //Log.v("MG", "getNormalizedProbability at labelIndex = " + labelIndex);
    return labelProbArray[0][labelIndex][0][0];
  }

  @Override
  protected ArrayList<Recognition> runInference() {
    // MG: This is where the model output should be catched
    Log.v("MG", "Starting to run inference");
    tflite.run(imgData, labelProbArray);

    getPoses(labelProbArray, poses_x, poses_y);

    // Fill results in ArrayList:
    ArrayList<Recognition> recognitions = new ArrayList<Recognition>();
    RectF pose_location = null;
    Float confidence = new Float(0.9);
    for (int k = 0; k < 17; ++k) {
      pose_location = new RectF(poses_x[k], poses_y[k],poses_x[k]+200,poses_x[k]+200);
      recognitions.add(new Recognition("Nose", "NoseTitle", confidence,pose_location ));
    }
    return recognitions;
  }


  void getPoses(float[][][][] labelProbArray, float[] poses_x, float[] poses_y){
    for(int k = 0; k < 17; k++){
      tmp_max[k] = labelProbArray[0][0][0][k];
      poses_x[k] = 0;
      poses_y[k] = 0;
    }


    for(int k = 0; k < 17; k++){ // loop over classes
      for(int i = 0; i < 23; i++){ // loop over x-dims
        for(int j = 0; j < 17; j++){ // loop over y-dims




          if (labelProbArray[0][i][j][k] > tmp_max[k]){
            Log.v("POSE", "old max[k] = " + tmp_max[k] + ",\t new max = " + labelProbArray[0][i][j][k] + ", at (k,i,j) = (" + k + "," + i + "," + j + ")" + " pose(x,y) = " + (int) poses_x[k] + "," + (int) poses_y[k]);
            tmp_max[k] = labelProbArray[0][i][j][k];
            poses_x[k] = i * output_stride;
            poses_y[k] = j * output_stride;
            //Log.v("POSE", " pose(x,y) = " + (int) poses_x[k] + "," + (int) poses_y[k]);
          }

        }
      }
    }
  }


}
