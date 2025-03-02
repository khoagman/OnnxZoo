using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxZoo.Extensions;
using OpenCvSharp;
using System.Collections.Concurrent;
using System.Diagnostics;

namespace OnnxZoo.OBB;
public class v8OBBPredictor : PredictorBase<Prediction>, IPredictor<Prediction> {
    public Stopwatch Stopwatch = new Stopwatch();
    public static IPredictor<Prediction> Create(string modelPath, string[] labels, bool useCuda = false) => new v8OBBPredictor(modelPath, labels, useCuda);
    private v8OBBPredictor(string modelPath, string[] labels, bool useCuda = false) : base(modelPath, labels, useCuda) { }

    private Prediction[] Pred_32(DenseTensor<float> tensor, Mat image) {
        var (w, h) = (image.Cols, image.Rows);
        var (xGain, yGain) = (InputWidth / (float)w, InputHeight / (float)h);
        var (xPad, yPad) = ((InputWidth - w * xGain) / 2, (InputHeight - h * yGain) / 2);

        int totalI = tensor.Dimensions[0];
        int totalJ = (int)(tensor.Length / (tensor.Dimensions[1] * tensor.Dimensions[0]));
        int outputDim = OutputDim;

        var resultLock = new object();
        var resultList = new List<Prediction>();

        Parallel.For(0, totalI * totalJ, () => new List<Prediction>(), (index, state, localList) =>
        {
            int i = index / totalJ;
            int j = index % totalJ;

            float xCenter = tensor[i, 0, j];
            float yCenter = tensor[i, 1, j];
            float boxWidth = tensor[i, 2, j];
            float boxHeight = tensor[i, 3, j];
            float angle = tensor[i, outputDim - 1, j];

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

            // Pre-calculate box parameters
            var center = new Point2f((xMin + xMax) / 2, (yMin + yMax) / 2);
            var size = new Size2f(finalWidth, finalHeight);
            var rotatedRect = new RotatedRect(center, size, angle);

            // Process classes
            for (int l = 0; l < outputDim - 5; l++) {
                float confidence = tensor[i, 4 + l, j];
                if (confidence < Confidence) continue;

                localList.Add(new Prediction {
                    Label = Labels[l],
                    Score = confidence,
                    Box = rotatedRect
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

    private Prediction[] Pred_32_Archived(DenseTensor<float> tensor, Mat image) {
        var result = new ConcurrentBag<Prediction>();

        var (w, h) = (image.Cols, image.Rows);
        var (xGain, yGain) = (InputWidth / (float)w, InputHeight / (float)h);
        var (xPad, yPad) = ((InputWidth - w * xGain) / 2, (InputHeight - h * yGain) / 2);

        Parallel.For(0, tensor.Dimensions[0], i => {
            //divide total length by the elements per prediction
            Parallel.For(0, (int)(tensor.Length / tensor.Dimensions[1]), j => {

                float xMin = ((tensor[i, 0, j] - tensor[i, 2, j] / 2) - xPad) / xGain; // unpad bbox tlx to original
                float yMin = ((tensor[i, 1, j] - tensor[i, 3, j] / 2) - yPad) / yGain; // unpad bbox tly to original
                float xMax = ((tensor[i, 0, j] + tensor[i, 2, j] / 2) - xPad) / xGain; // unpad bbox brx to original
                float yMax = ((tensor[i, 1, j] + tensor[i, 3, j] / 2) - yPad) / yGain; // unpad bbox bry to original
                float angle = tensor[i, OutputDim - 1, j];

                xMin = Utils.Clamp(xMin, 0, w - 0); // clip bbox tlx to boundaries
                yMin = Utils.Clamp(yMin, 0, h - 0); // clip bbox tly to boundaries
                xMax = Utils.Clamp(xMax, 0, w - 1); // clip bbox brx to boundaries
                yMax = Utils.Clamp(yMax, 0, h - 1); // clip bbox bry to boundaries

                Parallel.For(4, OutputDim - 1, l => {
                    var pred = tensor[i, l, j];

                    //skip low confidence values
                    if (pred < Confidence) return;

                    var prediction = new Prediction {
                        Label = Labels[l],
                        Score = pred,
                        Box = new RotatedRect(new Point2f((xMin + xMax) / 2, (yMin + yMax) / 2), new Size2f(xMax - xMin, yMax - yMin), angle)
                    }; 

                    result.Add(prediction);
                });
            });
        });
        return result.ToArray();
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
                float angle = tensor[i, 4, j].ToFloat();

                xMin = Utils.Clamp(xMin, 0, w - 0); // clip bbox tlx to boundaries
                yMin = Utils.Clamp(yMin, 0, h - 0); // clip bbox tly to boundaries
                xMax = Utils.Clamp(xMax, 0, w - 1); // clip bbox brx to boundaries
                yMax = Utils.Clamp(yMax, 0, h - 1); // clip bbox bry to boundaries

                Parallel.For(0, OutputDim - 5, l => {
                    var pred = tensor[i, 5 + l, j];

                    //skip low confidence values
                    if (pred.ToFloat() < Confidence) return;

                    var prediction = new Prediction {
                        Label = Labels[l],
                        Score = pred.ToFloat(),
                        Box = new RotatedRect(new Point2f(xMin, yMin), new Size2f(xMax - xMin, yMax - yMin), angle)
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
            Stopwatch.Restart();
            var result = Pred_32(output32, image);
            Stopwatch.Stop();
            Console.WriteLine($"Extraction time: {Stopwatch.Elapsed.TotalSeconds}");
            return PostProcessing.Suppress(result, Overlap);
        }
    }
}
