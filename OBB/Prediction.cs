using OpenCvSharp;

namespace OnnxZoo.OBB;
public class Prediction {
    public Label? Label { get; init; }
    public float Score { get; init; }
    public RotatedRect Box { get; init; }
}
