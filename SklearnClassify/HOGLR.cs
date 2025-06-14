using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxZoo.Extensions;
using OpenCvSharp;
using System.Collections.Concurrent;
using System.Diagnostics;

namespace OnnxZoo.SklearnClassify;
public class HOGLRPredictor : PredictorBase<Prediction>, IPredictor<Prediction> {
    public static IPredictor<Prediction> Create(string modelPath, string[] labels, bool useCuda = false) => new HOGLRPredictor(modelPath, labels, useCuda);
    private HOGLRPredictor(string modelPath, string[] labels, bool useCuda = false) : base(modelPath, labels, useCuda) { }

    private Prediction[] Pred_32(DenseTensor<float> tensor, Mat image) {
        var resultLock = new object();
        var resultList = new List<Prediction>();

        for (int i = 0; i < tensor.Dimensions[1]; i++) {
            float confidence = tensor[0, i];
            var label = Labels[i];
            resultList.Add(new Prediction {
                Label = label,
                Score = confidence
            });
        }
        return resultList.ToArray();
    }

    private Prediction[] Pred_16(DenseTensor<Float16> tensor, Mat image) {
        var resultLock = new object();
        var resultList = new List<Prediction>();

        Console.WriteLine(tensor.Dimensions[0]);
        return resultList.ToArray();
    }

    override public Prediction[] Predict(Mat image) {
        if (isFP16) {
            var output16 = InferenceHOG<Float16>(image);
            var result = Pred_16(output16, image);

            return result;
        }
        else {
            var output32 = InferenceHOG<float>(image);
            var result = Pred_32(output32, image);
            return PostProcessing.HighestOne(result);
        }
    }
}

