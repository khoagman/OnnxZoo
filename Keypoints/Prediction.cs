namespace OnnxZoo.Keypoints;

public class Prediction: OnnxZoo.ObjectDetection.Prediction {
    public int[]? PointsX { get; init; }
    public int[]? PointsY { get; init; }
    public float[]? KeypointScores { get; init; }
}
