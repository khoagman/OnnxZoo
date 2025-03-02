using System.Diagnostics;
using ByteTrack;
using OnnxZoo.Extensions;

namespace OnnxZoo.Keypoints;

public static class PostProcessing {
    public static Prediction[] Suppress(Prediction[] predictions, float overlapThresh) {
        Stopwatch stopwatch = new Stopwatch();
        stopwatch.Restart();
        if (predictions.Length == 0) return Array.Empty<Prediction>();

        Array.Sort(predictions, (a, b) => b.Score.CompareTo(a.Score));

        var keep = new List<int>();
        var suppress = new bool[predictions.Length];

        var areas = new float[predictions.Length];
        for (int i = 0; i < predictions.Length; i++) {
            var box = predictions[i].Coordinate;
            areas[i] = (box[2] - box[0]) * (box[3] - box[1]);
        }

        for (int i = 0; i < predictions.Length; i++) {
            if (suppress[i]) continue;

            keep.Add(i);
            var currentBox = predictions[i].Coordinate;

            float x1 = currentBox[0];
            float y1 = currentBox[1];
            float x2 = currentBox[2];
            float y2 = currentBox[3];

            for (int j = i + 1; j < predictions.Length; j++) {
                if (suppress[j]) continue;

                var nextBox = predictions[j].Coordinate;

                float xx1 = Math.Max(x1, nextBox[0]);
                float yy1 = Math.Max(y1, nextBox[1]);
                float xx2 = Math.Min(x2, nextBox[2]);
                float yy2 = Math.Min(y2, nextBox[3]);

                float w = Math.Max(0f, xx2 - xx1);
                float h = Math.Max(0f, yy2 - yy1);
                float intersection = w * h;

                float iou = intersection / (areas[i] + areas[j] - intersection);

                if (iou > overlapThresh) {
                    suppress[j] = true;
                }
            }
        }

        var result = new Prediction[keep.Count];
        for (int i = 0; i < keep.Count; i++) {
            result[i] = predictions[keep[i]];
        }
        stopwatch.Stop();
        Console.WriteLine($"Suppress time: {stopwatch.Elapsed.TotalMilliseconds} ms");
        return result;
    }
}

