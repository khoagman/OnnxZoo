using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxZoo.Extensions;
using OpenCvSharp;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;

namespace OnnxZoo.Classification;
public class Effv2Predictor : PredictorBase<Prediction>, IPredictor<Prediction> {
    public static IPredictor<Prediction> Create(string modelPath, string[] labels, bool useCuda = false) => new Effv2Predictor(modelPath, labels, useCuda);
    private Effv2Predictor(string modelPath, string[] labels, bool useCuda = false) : base(modelPath, labels, useCuda) { }

    private Prediction[][] Pred_32(DenseTensor<float> tensor) {
        var resultLock = new object();
        var resultList = new List<Prediction[]>();

        for (int i = 0; i < tensor.Dimensions[0]; i++) {
            var result = new List<Prediction>();
            for (int j = 0; j < tensor.Dimensions[1]; j++) {
                float confidence = tensor[0, j];
                //if (confidence < Confidence) continue;
                var label = Labels[j];
                result.Add(new Prediction {
                    Label = label,
                    Score = confidence
                });
            }
            resultList.Add(result.ToArray());
        }
        return resultList.ToArray();
    }

    private Prediction[] Pred_16(DenseTensor<Float16> tensor, Mat image) {
        var resultLock = new object();
        var resultList = new List<Prediction>();
        for (int i = 0; i < tensor.Dimensions[0]; i++) {
            float confidence = (float)tensor[i, 0];
            if (confidence < Confidence) continue;
            var label = Labels[0];
            resultList.Add(new Prediction {
                Label = label,
                Score = confidence
            });
        }
        return resultList.ToArray();
    }

    override public Prediction[] Predict(Mat image) {
        if (isFP16) {
            var output16 = InferenceClass<Float16>(image);
            var result = Pred_16(output16, image);
            return PostProcessing.HighestOne(result);
        }
        else {
            var output32 = InferenceClass<float>(image);
            var result = Pred_32(output32);
            return PostProcessing.HighestOne(result[0]);
        }
    }

    override public Prediction[] BatchPredict(List<Mat> images) {
        Debug.WriteLine("Start BatchInference");
        var output32 = BatchInference<float>(images);
        Debug.WriteLine("End BatchInference");
        var results = Pred_32(output32);
        return PostProcessing.BatchHighest(results);
    }
}

