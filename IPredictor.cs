using OpenCvSharp;

namespace OnnxZoo;

public interface IPredictor<TPrediction>: IDisposable {
    string? InputCol { get; }
    string? OutputCol { get; }

    int InputHeight { get; }
    int InputWidth { get; }

    int OutputDim { get; }

    Label[] getLabels();

    TPrediction[] Predict(Mat img);
}