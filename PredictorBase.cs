using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

using OnnxZoo.Extensions;
using System.Diagnostics;

namespace OnnxZoo; 
abstract public class PredictorBase<TPrediction>: IPredictor<TPrediction> {
    protected readonly InferenceSession _inferenceSession;
    protected string[] modelOutputs;
    public bool isFP16 = false;
    private Stopwatch Stopwatch = new Stopwatch();

    protected Label[] Labels { get; set; } = new Label[] { };

    public float Confidence { get; set; } = 0.5f;
    public float MulConfidence { get; set; } = 0.5f;
    public float Overlap { get; set; } = 0.5f;

    public string? InputCol { get; protected set; }
    public string? OutputCol { get; protected set; }
    public int InputHeight { get; protected set; }
    public int InputWidth { get; protected set; }
    public int OutputDim { get; protected set; }
    public int NumClasses { get; protected set; }
    public bool UseDetect { get; set; }

    private static SessionOptions CreateSessionOptions(bool useCuda)
    {
        var sessionOptions = new SessionOptions();
        var env = OrtEnv.Instance();

        var providers = env.GetAvailableProviders();
        foreach (var provider in providers)
        {
            Console.WriteLine($"- {provider}");
        }

        if (providers.Contains("CUDAExecutionProvider"))
        {
            Console.WriteLine("CUDA is available!");
        }
        else
        {
            Console.WriteLine("CUDA is NOT available. Check your CUDA installation.");
        }

        if (useCuda)
        {
            //sessionOptions.AppendExecutionProvider_CUDA(0);
            sessionOptions = SessionOptions.MakeSessionOptionWithCudaProvider(0);
            Console.WriteLine($"InferenceSession created successfully. Using CUDA: {useCuda}");
        }

        sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;

        sessionOptions.ExecutionMode = ExecutionMode.ORT_PARALLEL;

        return sessionOptions;
    }

    public void Dispose() {
        _inferenceSession.Dispose();
    }
    protected void GetInputDetails() {
        InputCol = _inferenceSession.InputMetadata.Keys.First();
        InputHeight = _inferenceSession.InputMetadata[InputCol].Dimensions[2];
        InputWidth = _inferenceSession.InputMetadata[InputCol].Dimensions[3];
    }

    protected virtual void GetOutputDetails() {
        OutputCol = _inferenceSession.OutputMetadata.Keys.First();
        modelOutputs = _inferenceSession.OutputMetadata.Keys.ToArray();
        OutputDim = _inferenceSession.OutputMetadata[modelOutputs[0]].Dimensions[1];
        UseDetect = !(modelOutputs.Any(x => x == "score"));

        foreach (var output in _inferenceSession.OutputMetadata)
        {
            if (output.Value.ElementType.ToString() == "Microsoft.ML.OnnxRuntime.Float16")
            {
                Console.WriteLine($"FP16");
                isFP16 = true;
            }
            else
            {
                Console.WriteLine($"FP32");
            }
        }
    }

    protected PredictorBase(string modelPath, string[]? labels = null, bool useCuda = false) {
        //if (useCuda) {
        //    _inferenceSession = new InferenceSession(modelPath, SessionOptions.MakeSessionOptionWithCudaProvider());
        //}
        //else {
        //    _inferenceSession = new InferenceSession(modelPath);
        //}

        _inferenceSession = new InferenceSession(modelPath, CreateSessionOptions(useCuda));

        GetInputDetails();
        GetOutputDetails();

        if (labels != null) {
            UseCustomLabels(labels);
        }
        else {
            UseCoCoLabels();
        }
        NumClasses = Labels.Length;
    }

    protected void UseCoCoLabels() {
        var s = new string[] { "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush" };
        UseCustomLabels(s);
    }

    protected void UseCustomLabels(string[] labels) {
        Labels = labels.Select((s, i) => new { i, s }).ToList()
            .Select(x => new Label() {
                Id = x.i,
                Name = x.s
            }).ToArray();
    }

    protected virtual DenseTensor<T> Inference<T>(Mat img) where T : struct {
        Mat resized = null;

        if (img.Rows != InputHeight || img.Cols != InputWidth) {
            resized = Utils.Resize(img, InputWidth, InputHeight);
        }
        else {
            resized = img;
        }

        DenseTensor<float> float32Data = Utils.ExtractPixels(resized) as DenseTensor<float>;
        if (float32Data == null)
            throw new Exception("Failed to extract pixels as DenseTensor<float>.");

        float[] floatArray = float32Data.Buffer.Span.ToArray();

        DenseTensor<T> tensor;

        if (typeof(T) == typeof(Float16)) {
            Float16[] float16Array = Array.ConvertAll(floatArray, f => (Float16)f);
            tensor = new DenseTensor<T>(float16Array as T[], new[] { 1, 3, resized.Rows, resized.Cols });
        }
        else if (typeof(T) == typeof(float)) {
            tensor = new DenseTensor<T>(floatArray as T[], new[] { 1, 3, resized.Rows, resized.Cols });
        }
        else {
            throw new NotSupportedException("Inference only supports float or BFloat16.");
        }

        if (tensor == null)
            throw new Exception("Tensor creation failed. Ensure the data conversion is correct.");

        var inputs = new List<NamedOnnxValue> {
            NamedOnnxValue.CreateFromTensor(InputCol, tensor)
        };

        if (inputs == null)
            throw new Exception("inputs creation failed. Ensure the data conversion is correct.");

        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = _inferenceSession.Run(inputs);

        DisposableNamedOnnxValue output = results.FirstOrDefault(x => modelOutputs.Contains(x.Name));

        if (output?.Value is not DenseTensor<T> outputTensor)
            throw new Exception("Inference output is null or invalid.");

        return outputTensor;
    }

    public Label[] getLabels() => Labels;

    public virtual TPrediction[] Predict(Mat image) {
        throw new NotImplementedException();
    }
}
