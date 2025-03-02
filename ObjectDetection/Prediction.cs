namespace OnnxZoo.ObjectDetection;

public class Prediction {
    public Label? Label { get; init; }
    public float Score { get; init; }
    public int[]? Coordinate { get; init; } = new int[4];
}
