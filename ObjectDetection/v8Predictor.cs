﻿using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxZoo.Extensions;
using OpenCvSharp;
using System.Collections.Concurrent;
using System.Diagnostics;

namespace OnnxZoo.ObjectDetection; 
public class v8Predictor : PredictorBase<Prediction>, IPredictor<Prediction> {
    public static IPredictor<Prediction> Create(string modelPath, string[] labels, bool useCuda = false) => new v8Predictor(modelPath, labels, useCuda);
    protected v8Predictor(string modelPath, string[] labels, bool useCuda = false) : base(modelPath, labels, useCuda) { }

    private Prediction[] Pred_32(DenseTensor<float> tensor, Mat image) {
        var (w, h) = (image.Cols, image.Rows);
        var (xGain, yGain) = (InputWidth / (float)w, InputHeight / (float)h);
        var (xPad, yPad) = ((InputWidth - w * xGain) / 2, (InputHeight - h * yGain) / 2);

        int totalI = tensor.Dimensions[0];
        int totalJ = (int)(tensor.Length / (tensor.Dimensions[1] * tensor.Dimensions[0]));
        int outputDim = OutputDim;

        var resultLock = new object();
        var resultList = new List<Prediction>();

        Parallel.For(0, totalI * totalJ, () => new List<Prediction>(), (index, state, localList) => {
            int i = index / totalJ;
            int j = index % totalJ;

            float xCenter = tensor[i, 0, j];
            float yCenter = tensor[i, 1, j];
            float boxWidth = tensor[i, 2, j];
            float boxHeight = tensor[i, 3, j];

            // Calculate coordinates
            float xMin = (xCenter - boxWidth / 2 - xPad) / xGain;
            float yMin = (yCenter - boxHeight / 2 - yPad) / yGain;
            float xMax = (xCenter + boxWidth / 2 - xPad) / xGain;
            float yMax = (yCenter + boxHeight / 2 - yPad) / yGain;

            // Clamp coordinates
            xMin = Math.Clamp(xMin, 0, w - 0);
            yMin = Math.Clamp(yMin, 0, h - 0);
            xMax = Math.Clamp(xMax, 0, w - 1);
            yMax = Math.Clamp(yMax, 0, h - 1);

            // Skip invalid boxes
            float finalWidth = xMax - xMin;
            float finalHeight = yMax - yMin;
            if (finalWidth <= 0 || finalHeight <= 0) return localList;

            // Process classes
            for (int l = 0; l < OutputDim - 4; l++) {
                float confidence = tensor[i, l + 4, j];
                if (confidence < Confidence) continue;

                localList.Add(new Prediction {
                    Label = Labels[l],
                    Score = confidence,
                    Coordinate = new int[] { (int)xMin, (int)yMin, (int)xMax, (int)yMax }
                });
            }

            return localList;
        },
        localList => {
            lock (resultLock) {
                resultList.AddRange(localList);
            }
        });

        return resultList.ToArray();
    }

    private Prediction[] Pred_16(DenseTensor<Float16> tensor, Mat image) {
        var result = new ConcurrentBag<Prediction>();

        var (w, h) = (image.Cols, image.Rows);
        var (xGain, yGain) = (InputWidth / (float)w, InputHeight / (float)h);
        var (xPad, yPad) = ((InputWidth - w * xGain) / 2, (InputHeight - h * yGain) / 2);

        Parallel.For(0, tensor.Dimensions[0], i => {
            //divide total length by the elements per prediction
            Parallel.For(0, (int)(tensor.Length / tensor.Dimensions[1]), j => {
                float xMin = ((tensor[i, 0, j].ToFloat() - tensor[i, 2, j].ToFloat() / 2) - xPad) / xGain; // unpad bbox tlx to original
                float yMin = ((tensor[i, 1, j].ToFloat() - tensor[i, 3, j].ToFloat() / 2) - yPad) / yGain; // unpad bbox tly to original
                float xMax = ((tensor[i, 0, j].ToFloat() + tensor[i, 2, j].ToFloat() / 2) - xPad) / xGain; // unpad bbox brx to original
                float yMax = ((tensor[i, 1, j].ToFloat() + tensor[i, 3, j].ToFloat() / 2) - yPad) / yGain; // unpad bbox bry to original

                xMin = Utils.Clamp(xMin, 0, w - 0); // clip bbox tlx to boundaries
                yMin = Utils.Clamp(yMin, 0, h - 0); // clip bbox tly to boundaries
                xMax = Utils.Clamp(xMax, 0, w - 1); // clip bbox brx to boundaries
                yMax = Utils.Clamp(yMax, 0, h - 1); // clip bbox bry to boundaries

                Parallel.For(0, OutputDim - 4, l => {
                    var pred = tensor[i, 4 + l, j];

                    //skip low confidence values
                    if (pred.ToFloat() < Confidence) return;

                    var prediction = new Prediction {
                        Label = Labels[l],
                        Score = pred.ToFloat(),
                        Coordinate = new int[] { (int)xMin, (int)yMin, (int)xMax, (int)yMax }
                    };

                    result.Add(prediction);
                });
            });
        });
        return result.ToArray();
    }

    override public Prediction[] Predict(Mat image) {
        if (isFP16) {
            var output16 = Inference<Float16>(image);
            var result = Pred_16(output16, image);
            return PostProcessing.Suppress(result, Overlap);
        }
        else {
            var output32 = Inference<float>(image);
            var result = Pred_32(output32, image);
          return PostProcessing.Suppress(result, Overlap);
        }
    }
    override public Prediction[] BatchPredict(List<Mat> images) {
        throw new NotImplementedException();
    }
}
