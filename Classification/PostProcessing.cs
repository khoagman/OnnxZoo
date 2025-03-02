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

        for (int i = 0; i < predictions.Length; i++) {
            if (predictions[i].Score > maxScore) {
                maxScore = predictions[i].Score;
                maxScoreIndex = i;
            }
        }

        results[0] = predictions[maxScoreIndex];

        return results;
    }
}
