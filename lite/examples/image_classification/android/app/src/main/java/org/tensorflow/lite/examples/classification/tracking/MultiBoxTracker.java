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

package org.tensorflow.lite.examples.classification.tracking;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Cap;
import android.graphics.Paint.Join;
import android.graphics.Paint.Style;
import android.graphics.Point;
import android.graphics.PointF;
import android.graphics.Rect;
import android.graphics.RectF;
import android.text.TextUtils;
import android.util.Log;
import android.util.Pair;
import android.util.TypedValue;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import org.tensorflow.lite.examples.classification.env.BorderedText;
import org.tensorflow.lite.examples.classification.env.ImageUtils;
import org.tensorflow.lite.examples.classification.env.Logger;
import org.tensorflow.lite.examples.classification.tflite.Classifier.Recognition;

/** A tracker that handles non-max suppression and matches existing objects to new detections. */
public class MultiBoxTracker {
  private static final float TEXT_SIZE_DIP = 18;
  private static final float MIN_SIZE = 16.0f;
  private static final int[] COLORS = {
    Color.BLUE,
    Color.RED,
    Color.GREEN,
    Color.YELLOW,
    Color.CYAN,
    Color.MAGENTA,
    Color.WHITE,
    Color.parseColor("#55FF55"),
    Color.parseColor("#FFA500"),
    Color.parseColor("#FF8888"),
    Color.parseColor("#AAAAFF"),
    Color.parseColor("#FFFFAA"),
    Color.parseColor("#55AAAA"),
    Color.parseColor("#AA33AA"),
    Color.parseColor("#0D0068")
  };
  final List<Pair<Float, RectF>> screenRects = new LinkedList<Pair<Float, RectF>>();
  private final Logger logger = new Logger();
  private final Queue<Integer> availableColors = new LinkedList<Integer>();
  private final List<TrackedRecognition> trackedObjects = new LinkedList<TrackedRecognition>();
  private final Paint boxPaint = new Paint();
  private final float textSizePx;
  private final BorderedText borderedText;
  private Matrix frameToCanvasMatrix;
  private int frameWidth;
  private int frameHeight;
  private int sensorOrientation;
  private List<Recognition> poses = null;

  private float minPoseConf = 0.1f;
  private float minPartConf = 0.5f;


  private Keypoint nose = new Keypoint();
  private Keypoint leftEye = new Keypoint();
  private Keypoint rightEye = new Keypoint();
  private Keypoint leftEar = new Keypoint();
  private Keypoint rightEar = new Keypoint();
  private Joint shoulder_left = new Joint();
  private Joint shoulder_right = new Joint();
  private Joint elbow_left = new Joint();
  private Joint elbow_right  = new Joint();
  private Joint wrist_left = new Joint();
  private Joint wrist_right = new Joint();
  private Joint hip_left = new Joint();
  private Joint hip_right = new Joint();
  private Joint knee_left = new Joint();
  private Joint knee_right = new Joint();
  private Joint ankle_left = new Joint();
  private Joint ankle_right = new Joint();


  public MultiBoxTracker(final Context context) {
    for (final int color : COLORS) {
      availableColors.add(color);
    }

    boxPaint.setColor(Color.RED);
    boxPaint.setStyle(Style.STROKE);
    boxPaint.setStrokeWidth(10.0f);
    boxPaint.setStrokeCap(Cap.ROUND);
    boxPaint.setStrokeJoin(Join.ROUND);
    boxPaint.setStrokeMiter(100);

    textSizePx =
        TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, context.getResources().getDisplayMetrics());
    borderedText = new BorderedText(textSizePx);
  }

  public synchronized void setFrameConfiguration(
      final int width, final int height, final int sensorOrientation) {
    frameWidth = width;
    frameHeight = height;
    this.sensorOrientation = sensorOrientation;
  }

  public synchronized void drawDebug(final Canvas canvas) {
    final Paint textPaint = new Paint();
    textPaint.setColor(Color.WHITE);
    textPaint.setTextSize(60.0f);

    final Paint boxPaint = new Paint();
    boxPaint.setColor(Color.RED);
    boxPaint.setAlpha(200);
    boxPaint.setStyle(Style.STROKE);

    for (final Pair<Float, RectF> detection : screenRects) {
      final RectF rect = detection.second;
      canvas.drawRect(rect, boxPaint);
      canvas.drawText("" + detection.first, rect.left, rect.top, textPaint);
      borderedText.drawText(canvas, rect.centerX(), rect.centerY(), "" + detection.first);
    }
  }

  public synchronized void trackResults(final List<Recognition> results, final long timestamp) {
    logger.i("Processing %d results from %d", results.size(), timestamp);
    processResults(results);
  }

  private Matrix getFrameToCanvasMatrix() {
    return frameToCanvasMatrix;
  }

//  public synchronized void draw(final Canvas canvas) {
//    final boolean rotated = sensorOrientation % 180 == 90;
////    final boolean rotated = false;
//    final float multiplier =
//        Math.min(
//            canvas.getHeight() / (float) (rotated ? frameWidth : frameHeight),
//            canvas.getWidth() / (float) (rotated ? frameHeight : frameWidth));
//    frameToCanvasMatrix =
//        ImageUtils.getTransformationMatrix(
//            frameWidth,
//            frameHeight,
//            (int) (multiplier * (rotated ? frameHeight : frameWidth)),
//            (int) (multiplier * (rotated ? frameWidth : frameHeight)),
//            sensorOrientation,
//            false);
//    for (final TrackedRecognition recognition : trackedObjects) {
//      final RectF trackedPos = new RectF(recognition.location);
//
//      getFrameToCanvasMatrix().mapRect(trackedPos);
//      boxPaint.setColor(recognition.color);
//
//      float cornerSize = Math.min(trackedPos.width(), trackedPos.height()) / 8.0f;
//      canvas.drawRoundRect(trackedPos, cornerSize, cornerSize, boxPaint);
//
////      final String labelString =
////          !TextUtils.isEmpty(recognition.title)
////              ? String.format("%s %.2f", recognition.title, (100 * recognition.detectionConfidence))
////              : String.format("%.2f", (100 * recognition.detectionConfidence));
////      //            borderedText.drawText(canvas, trackedPos.left + cornerSize, trackedPos.top,
////      // labelString);
////      borderedText.drawText(
////          canvas, trackedPos.left + cornerSize, trackedPos.top, labelString + "%", boxPaint);
//    }
//  }

  public synchronized void setPoses(List<Recognition> resultPoses){
    poses = resultPoses;
  }

  public synchronized void draw_single_box(final Canvas canvas) {
    if (poses == null){
      return;
    }
    Log.v("RECT","draw_single_box");
    final boolean rotated = sensorOrientation % 180 == 90;
    final float multiplier =
            Math.min(
                    canvas.getHeight() / (float) (rotated ? frameWidth : frameHeight),
                    canvas.getWidth() / (float) (rotated ? frameHeight : frameWidth));
    frameToCanvasMatrix =
            ImageUtils.getTransformationMatrix(
                    frameWidth,
                    frameHeight,
                    (int) (multiplier * (rotated ? frameHeight : frameWidth)),
                    (int) (multiplier * (rotated ? frameWidth : frameHeight)),
                    sensorOrientation,
                    false);

    final RectF trackedPos = new RectF(0,0,640,480);
    getFrameToCanvasMatrix().mapRect(trackedPos);
    boxPaint.setColor(Color.RED);
    //float cornerSize = Math.min(trackedPos.width(), trackedPos.height()) / 8.0f;
    float cornerSize = 50;
    canvas.drawRoundRect(trackedPos, cornerSize, cornerSize, boxPaint);

    Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
    paint.setStyle(Paint.Style.STROKE);
    paint.setColor(Color.GREEN);
    paint.setStrokeWidth(10);

    Paint paint_red = new Paint(Paint.ANTI_ALIAS_FLAG);
    paint_red.setStyle(Paint.Style.STROKE);
    paint_red.setColor(Color.RED);
    paint_red.setStrokeWidth(3);

//    Rect rec2 = new Rect(10, 10, 1004, 2004);
//    canvas.drawRect(rec2,paint);

    float radius = 10;
    float scale = 2.0f;
    float [] mPoints = new float[34];
    float [] confidences = new float[17];
    float x = 0.0f, y = 0.0f;
    paint.setStyle(Paint.Style.FILL);
    int counter = 0;
    for (Recognition pose : poses){
      y = pose.getLocation().top * scale; // MG: "top" ironically means x-dim
      x = pose.getLocation().left * scale; // MG: "left" ironically means y-dim
      confidences[counter] = pose.getConfidence();
      mPoints[counter * 2] = x;
      mPoints[counter * 2 + 1] = y;
      counter ++;
      // Log.v("POSE","Counter = " + counter +  " x: " + x + " y: " + y);
    }


    // Draw network output resolution grid
    float [] gridPoints = new float[2];
    float output_stride = 16.0f;
    float x_max =  257;
    float y_max =  353;

//    counter = 0;
//    for (int i = 0; i < 17; i++){
//      for (int j = 0; j < 23; j ++){
//        //gridPoints[0] = j * output_stride + output_stride / 2.0f; // x-val
//        //gridPoints[1] = i * output_stride + output_stride / 2.0f; // y-val
//
//        gridPoints[0] = y_max - ( (float)j * output_stride + output_stride / 2.0f );
//        gridPoints[1] = x_max - ( (float)i * output_stride + output_stride / 2.0f );
//
//
//        canvas.drawCircle(gridPoints[1], gridPoints[0],4.0f,paint_red);
//        getFrameToCanvasMatrix().mapPoints(gridPoints);
//        canvas.drawCircle(gridPoints[1], gridPoints[0],4.0f,paint);
//      }
//    }




    getFrameToCanvasMatrix().mapPoints(mPoints);

    nose.confidence = confidences[0];
    nose.location.x = mPoints[0 * 2];
    nose.location.y = mPoints[0 * 2 + 1];
    leftEye.confidence = confidences[1];
    leftEye.location.x = mPoints[1 * 2];
    leftEye.location.y = mPoints[1 * 2 + 1];
    rightEye.confidence = confidences[2];
    rightEye.location.x = mPoints[2 * 2];
    rightEye.location.y = mPoints[2 * 2 + 1];
    leftEar.confidence = confidences[3];
    leftEar.location.x = mPoints[3 * 2];
    leftEar.location.y = mPoints[3 * 2 + 1];
    rightEar.confidence = confidences[4];
    rightEar.location.x = mPoints[4 * 2];
    rightEar.location.y = mPoints[4 * 2 + 1];

    shoulder_left.confidence = confidences[5];
    shoulder_left.location.x = mPoints[5 * 2];
    shoulder_left.location.y = mPoints[5 * 2 + 1];
    shoulder_right.confidence = confidences[6];
    shoulder_right.location.x = mPoints[6 * 2];
    shoulder_right.location.y = mPoints[6 * 2 + 1];

    elbow_left.confidence = confidences[7];
    elbow_left.location.x = mPoints[7 * 2];
    elbow_left.location.y = mPoints[7 * 2 + 1];
    elbow_right.confidence = confidences[8];
    elbow_right.location.x = mPoints[8 * 2];
    elbow_right.location.y = mPoints[8 * 2 + 1];

    wrist_left.confidence = confidences[9];
    wrist_left.location.x = mPoints[9 * 2];
    wrist_left.location.y = mPoints[9 * 2 + 1];
    wrist_right.confidence = confidences[10];
    wrist_right.location.x = mPoints[10 * 2];
    wrist_right.location.y = mPoints[10 * 2 + 1];

    hip_left.confidence = confidences[11];
    hip_left.location.x = mPoints[11 * 2];
    hip_left.location.y = mPoints[11 * 2 + 1];
    hip_right.confidence = confidences[12];
    hip_right.location.x = mPoints[12 * 2];
    hip_right.location.y = mPoints[12 * 2 + 1];

    knee_left.confidence = confidences[13];
    knee_left.location.x = mPoints[13 * 2];
    knee_left.location.y = mPoints[13 * 2 + 1];
    knee_right.confidence = confidences[14];
    knee_right.location.x = mPoints[14 * 2];
    knee_right.location.y = mPoints[14 * 2 + 1];

    ankle_left.confidence = confidences[15];
    ankle_left.location.x = mPoints[15 * 2];
    ankle_left.location.y = mPoints[15 * 2 + 1];
    ankle_right.confidence = confidences[16];
    ankle_right.location.x = mPoints[16 * 2];
    ankle_right.location.y = mPoints[16 * 2 + 1];



    // Face
    draw_keypoint(nose, paint, canvas);
    draw_keypoint(leftEye, paint, canvas);
    draw_keypoint(rightEye, paint, canvas);
    draw_keypoint(leftEar, paint, canvas);
    draw_keypoint(rightEar, paint, canvas);


    // Arms
    paint.setColor(Color.MAGENTA);
    draw_joint(shoulder_left, elbow_left, paint, canvas);
    draw_joint(elbow_left, wrist_left, paint, canvas);

    paint.setColor(Color.CYAN);
    draw_joint(shoulder_right, elbow_right, paint, canvas);
    draw_joint(elbow_right, wrist_right, paint, canvas);

    // Legs
    paint.setColor(Color.BLUE);
    draw_joint(hip_left, knee_left, paint, canvas);
    draw_joint(knee_left, ankle_left, paint, canvas);

    paint.setColor(Color.GREEN);
    draw_joint(hip_right, knee_right, paint, canvas);
    draw_joint(knee_right, ankle_right, paint, canvas);

    // Body
    paint.setColor(Color.YELLOW);
    draw_joint(shoulder_left, shoulder_right, paint, canvas);
    draw_joint(shoulder_left, hip_left, paint, canvas);
    draw_joint(shoulder_right, hip_right, paint, canvas);
    draw_joint(hip_left, hip_right, paint, canvas);

  }

  private void draw_joint(Joint joint_a, Joint joint_b, Paint paint, Canvas canvas){
      if(joint_a.confidence > minPartConf && joint_b.confidence > minPartConf){
          canvas.drawLine(joint_a.location.x,joint_a.location.y,joint_b.location.x,joint_b.location.y,paint);
      }
  }

  private void draw_keypoint(Keypoint keypoint, Paint paint, Canvas canvas){
    if(keypoint.confidence > minPoseConf){
      canvas.drawCircle(keypoint.location.x, keypoint.location.y,10.0f,paint);
    }
  }

  private void processResults(final List<Recognition> results) {
    Log.v("MG", "Results are not processed anymore (MultiBoxTracker)");
    final List<Pair<Float, Recognition>> rectsToTrack = new LinkedList<Pair<Float, Recognition>>();

    screenRects.clear();
    final Matrix rgbFrameToScreen = new Matrix(getFrameToCanvasMatrix());

    for (final Recognition result : results) {
      if (result.getLocation() == null) {
        continue;
      }
      final RectF detectionFrameRect = new RectF(result.getLocation());

      final RectF detectionScreenRect = new RectF();
      rgbFrameToScreen.mapRect(detectionScreenRect, detectionFrameRect);

      logger.v(
          "Result! Frame: " + result.getLocation() + " mapped to screen:" + detectionScreenRect);

      screenRects.add(new Pair<Float, RectF>(result.getConfidence(), detectionScreenRect));

      if (detectionFrameRect.width() < MIN_SIZE || detectionFrameRect.height() < MIN_SIZE) {
        logger.w("Degenerate rectangle! " + detectionFrameRect);
        continue;
      }

      rectsToTrack.add(new Pair<Float, Recognition>(result.getConfidence(), result));
    }

    if (rectsToTrack.isEmpty()) {
      logger.v("Nothing to track, aborting.");
      return;
    }

    trackedObjects.clear();
    for (final Pair<Float, Recognition> potential : rectsToTrack) {
      final TrackedRecognition trackedRecognition = new TrackedRecognition();
      trackedRecognition.detectionConfidence = potential.first;
      trackedRecognition.location = new RectF(potential.second.getLocation());
      trackedRecognition.title = potential.second.getTitle();
      trackedRecognition.color = COLORS[trackedObjects.size()];
      trackedObjects.add(trackedRecognition);

      if (trackedObjects.size() >= COLORS.length) {
        break;
      }
    }
  }

  private static class TrackedRecognition {
    RectF location;
    float detectionConfidence;
    int color;
    String title;
  }

  private class Joint {
    PointF location;
    float confidence;
    int color;
    String title;

    Joint(){
      location = new PointF();
    }
  }

  private class Keypoint {
    PointF location;
    float confidence;
    int color;
    String title;

    Keypoint(){
      location = new PointF();
    }
  }

}
