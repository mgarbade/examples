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

  public synchronized void draw(final Canvas canvas) {
    final boolean rotated = sensorOrientation % 180 == 90;
//    final boolean rotated = false;
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
    for (final TrackedRecognition recognition : trackedObjects) {
      final RectF trackedPos = new RectF(recognition.location);

      getFrameToCanvasMatrix().mapRect(trackedPos);
      boxPaint.setColor(recognition.color);

      float cornerSize = Math.min(trackedPos.width(), trackedPos.height()) / 8.0f;
      canvas.drawRoundRect(trackedPos, cornerSize, cornerSize, boxPaint);

//      final String labelString =
//          !TextUtils.isEmpty(recognition.title)
//              ? String.format("%s %.2f", recognition.title, (100 * recognition.detectionConfidence))
//              : String.format("%.2f", (100 * recognition.detectionConfidence));
//      //            borderedText.drawText(canvas, trackedPos.left + cornerSize, trackedPos.top,
//      // labelString);
//      borderedText.drawText(
//          canvas, trackedPos.left + cornerSize, trackedPos.top, labelString + "%", boxPaint);
    }
  }

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
    paint.setStrokeWidth(3);

    Paint paint_red = new Paint(Paint.ANTI_ALIAS_FLAG);
    paint_red.setStyle(Paint.Style.STROKE);
    paint_red.setColor(Color.RED);
    paint_red.setStrokeWidth(3);

//    Rect rec2 = new Rect(10, 10, 1004, 2004);
//    canvas.drawRect(rec2,paint);

    float radius = 10;
    float scale = (float) 1.0;
    float [] mPoints = new float[34];
    float x = 0, y = 0;
    paint.setStyle(Paint.Style.FILL);
    int counter = 0;
    for (Recognition pose : poses){
      y = pose.getLocation().top * scale;
      x = pose.getLocation().left * scale;
      mPoints[counter * 2] = x;
      mPoints[counter * 2 + 1] = y;
      counter ++;
      Log.v("POSE","Counter = " + counter +  " x: " + x + " y: " + y);
    }

    float [] gridPoints = new float[2];
    int output_stride = 32;
    counter = 0;
    for (int i = 0; i < 23; i++){ // y-dims
      for (int j = 0; j < 17; j ++){ // x-dims
        gridPoints[0] = j * output_stride;
        gridPoints[1] = i * output_stride;
//        canvas.drawCircle(gridPoints[1], gridPoints[0],3.0f,paint_red);
        getFrameToCanvasMatrix().mapPoints(gridPoints);
        canvas.drawCircle(gridPoints[1], gridPoints[0],4.0f,paint);
      }
    }




    getFrameToCanvasMatrix().mapPoints(mPoints);

    float shoulder_left_x = mPoints[5 * 2];
    float shoulder_left_y = mPoints[5 * 2 + 1];
    float shoulder_right_x = mPoints[6 * 2];
    float shoulder_right_y = mPoints[6 * 2 + 1];

    float elbow_left_x = mPoints[7 * 2];
    float elbow_left_y = mPoints[7 * 2 + 1];
    float elbow_right_x = mPoints[8 * 2];
    float elbow_right_y = mPoints[8 * 2 + 1];

    float wrist_left_x = mPoints[9 * 2];
    float wrist_left_y = mPoints[9 * 2 + 1];
    float wrist_right_x = mPoints[10 * 2];
    float wrist_right_y = mPoints[10 * 2 + 1];

    float hip_left_x = mPoints[11 * 2];
    float hip_left_y = mPoints[11 * 2 + 1];
    float hip_right_x = mPoints[12 * 2];
    float hip_right_y = mPoints[12 * 2 + 1];

    float knee_left_x = mPoints[13 * 2];
    float knee_left_y = mPoints[13 * 2 + 1];
    float knee_right_x = mPoints[14 * 2];
    float knee_right_y = mPoints[14 * 2 + 1];

    float ankle_left_x = mPoints[15 * 2];
    float ankle_left_y = mPoints[15 * 2 + 1];
    float ankle_right_x = mPoints[16 * 2];
    float ankle_right_y = mPoints[16 * 2 + 1];






    // Arms
    paint.setColor(Color.MAGENTA);
    canvas.drawLine(shoulder_left_x,shoulder_left_y,elbow_left_x,elbow_left_y,paint);
    canvas.drawLine(elbow_left_x,elbow_left_y,wrist_left_x,wrist_left_y,paint);
    paint.setColor(Color.CYAN);
    canvas.drawLine(shoulder_right_x,shoulder_right_y,elbow_right_x,elbow_right_y,paint);
    canvas.drawLine(elbow_right_x,elbow_right_y,wrist_right_x,wrist_right_y,paint);

    // Legs
    paint.setColor(Color.BLUE);
    canvas.drawLine(hip_left_x,hip_left_y,knee_left_x,knee_left_y,paint);
    canvas.drawLine(knee_left_x,knee_left_y,ankle_left_x,ankle_left_y,paint);
    paint.setColor(Color.GREEN);
    canvas.drawLine(hip_right_x,hip_right_y,knee_right_x,knee_right_y,paint);
    canvas.drawLine(knee_right_x,knee_right_y,ankle_right_x,ankle_right_y,paint);

    // Body
    paint.setColor(Color.YELLOW);
    canvas.drawLine(shoulder_left_x,shoulder_left_y,shoulder_right_x,shoulder_right_y,paint);
    canvas.drawLine(shoulder_left_x,shoulder_left_y,hip_left_x,hip_left_y,paint);
    canvas.drawLine(shoulder_right_x,shoulder_right_y,hip_right_x,hip_right_y,paint);
    canvas.drawLine(hip_left_x,hip_left_y,hip_right_x,hip_right_y,paint);


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
}
