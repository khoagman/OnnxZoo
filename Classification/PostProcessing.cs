using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OnnxZoo.Classification; 
public class PostProcessing {
    public static Prediction[] HighestOne(Prediction[] predictions) {
        if (predictions.Length == 0) return Array.Empty<Prediction>();

        var results = new Prediction[1];
        float maxScore = -1;
        int maxScoreIndex = 0;
        int secondMaxScoreIndex = 0;

        for (int i = 0; i < predictions.Length; i++) {
            if (predictions[i].Score > maxScore) {
                maxScore = predictions[i].Score;
                maxScoreIndex = i;
            }
        }
        maxScore = -1;
        for (int i = 0; i < predictions.Length; i++) {
            if (i == maxScoreIndex) continue;
            if (predictions[i].Score > maxScore) {
                maxScore = predictions[i].Score;
                secondMaxScoreIndex = i;
            }
        }

        if (predictions[maxScoreIndex].Label.Name == "D") {
            Debug.WriteLine($"D: {predictions[maxScoreIndex].Score}");
            Debug.WriteLine($"Not D: {predictions[secondMaxScoreIndex].Label.Name}");
            Debug.WriteLine($"Not D: {predictions[secondMaxScoreIndex].Score}");
        }

            results[0] = predictions[maxScoreIndex];

        return results;
    }

    public static Prediction[] BatchHighest(Prediction[][] predictions) {
        if (predictions.Length == 0) return Array.Empty<Prediction>();
        var results = new List<Prediction>();
        foreach (var prediction in predictions) {
            var result = HighestOne(prediction);
            results.AddRange(result);
        }
        return results.ToArray();
    }
}
