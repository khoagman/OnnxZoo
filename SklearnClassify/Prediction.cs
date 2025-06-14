using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OnnxZoo.SklearnClassify; 
public class Prediction {
    public Label? Label { get; init; }
    public float Score { get; init; }
}
