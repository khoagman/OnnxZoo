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
    public float Overlap { get; set; } = 0.6f;

    public string? InputCol { get; protected set; }
    public string? OutputCol { get; protected set; }
    public int InputHeight { get; protected set; }
    public int InputWidth { get; protected set; }
    public int VectorLength { get; protected set; }
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
        sessionOptions.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE;

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

    protected virtual DenseTensor<T> InferenceHOG<T>(Mat img) where T : struct {
        Mat resized = null;
        if (img.Rows != 64 || img.Cols != 32) {
            resized = Utils.Resize(img, 32, 64);
        }
        else {
            resized = img;
        }

        Mat gray = new Mat();
        if (resized.Channels() == 3)
            Cv2.CvtColor(resized, gray, ColorConversionCodes.BGR2GRAY);
        else
            gray = resized.Clone();

        Cv2.Resize(gray, gray, new Size(32, 64));

        Point[] locations = new Point[0];

        //float[] floatArray = hog.Compute(gray, new Size(8, 8), new Size(0, 0), locations);
        float[] floatArray =  Utils.ComputeHogFeatures(gray);

        DenseTensor<T> tensor;
        if (typeof(T) == typeof(Float16)) {
            Float16[] float16Array = Array.ConvertAll(floatArray, f => (Float16)f);
            tensor = new DenseTensor<T>(float16Array as T[], new[] { 1, VectorLength });
        }
        else if (typeof(T) == typeof(float)) {
            tensor = new DenseTensor<T>(floatArray as T[], new[] { 1, VectorLength });
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
        Console.WriteLine($"Inference completed in {Stopwatch.ElapsedMilliseconds} ms");
        var probabilitiesOutput = results.FirstOrDefault(x => x.Name == "probabilities");
        if (probabilitiesOutput?.Value is not DenseTensor<T> outputTensor)
            throw new Exception("Inference output is null or invalid.");
        return outputTensor;
    }

    protected virtual DenseTensor<T> InferenceClass<T>(Mat img) where T : struct {
        Mat resized = null;
        if (img.Rows != InputHeight || img.Cols != InputWidth) {
            resized = Utils.Resize(img, InputWidth, InputHeight);
        }
        else {
            resized = img;
        }

        // Trích xuất pixel
        DenseTensor<float> float32Data = Utils.ExtractPixels(resized) as DenseTensor<float>;
        if (float32Data == null)
            throw new Exception("Failed to extract pixels as DenseTensor<float>.");

        // Lấy dữ liệu dưới dạng mảng
        float[] floatArray = float32Data.Buffer.Span.ToArray();

        //Chuẩn hóa về[0, 1] trước
        //for (int i = 0; i < floatArray.Length; i++) {
        //    floatArray[i] = floatArray[i] / 255.0f;
        //}

        //Áp dụng chuẩn hóa ImageNet
        float[] means = new float[] { 0.485f, 0.456f, 0.406f };
        float[] stds = new float[] { 0.229f, 0.224f, 0.225f };

        for (int i = 0; i < floatArray.Length; i++) {
            int channelIdx = i % 3;
            floatArray[i] = (floatArray[i] - means[channelIdx]) / stds[channelIdx];
        }

        // Tạo tensor từ dữ liệu đã chuẩn hóa
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

    protected virtual DenseTensor<T> BatchInference<T>(List<Mat> images) where T : struct {
        if (images == null || images.Count == 0)
            throw new ArgumentException("Image list cannot be null or empty.");

        int batchSize = images.Count;
        Mat[] resizedImages = new Mat[batchSize];
        List<float[]> floatArrays = new List<float[]>(batchSize);
        
        try {
            for (int i = 0; i < batchSize; i++) {
                Mat img = images[i];
                if (img.Rows != InputHeight || img.Cols != InputWidth) {
                    resizedImages[i] = Utils.Resize(img, InputWidth, InputHeight);
                }
                else {
                    resizedImages[i] = img.Clone(); // Clone để tránh tham chiếu trực tiếp
                }

                // Kiểm tra kích thước sau resize
                if (resizedImages[i].Rows != InputHeight || resizedImages[i].Cols != InputWidth)
                    throw new Exception($"Ảnh {i} có kích thước sai sau resize.");

                DenseTensor<float> float32Data = Utils.ExtractPixels(resizedImages[i]) as DenseTensor<float>;
                if (float32Data == null)
                    throw new Exception($"Failed to extract pixels for image {i}.");

                floatArrays.Add(float32Data.Buffer.Span.ToArray());
            }
            // Tạo tensor batch
            int totalPixels = InputHeight * InputWidth * 3;
            float[] combinedFloatArray = new float[batchSize * totalPixels];
            for (int i = 0; i < batchSize; i++) {
                Array.Copy(floatArrays[i], 0, combinedFloatArray, i * totalPixels, totalPixels);
            }
            DenseTensor<T> tensor;
            var tensorShape = new[] { batchSize, 3, InputHeight, InputWidth };

            if (typeof(T) == typeof(Float16)) {
                Float16[] float16Array = Array.ConvertAll(combinedFloatArray, f => (Float16)f);
                tensor = new DenseTensor<T>(float16Array.Select(x => (T)(object)x).ToArray(), tensorShape);
            }
            else if (typeof(T) == typeof(float)) {
                tensor = new DenseTensor<T>(combinedFloatArray.Select(x => (T)(object)x).ToArray(), tensorShape);
            }
            else {
                throw new NotSupportedException("Chỉ hỗ trợ float hoặc Float16.");
            }
            // check tensor shape
            var inputs = new List<NamedOnnxValue>
            {
            NamedOnnxValue.CreateFromTensor(InputCol, tensor)
        };

            Debug.WriteLine($"tensor: {tensor.Dimensions[0]}x{tensor.Dimensions[1]}x{tensor.Dimensions[2]}x{tensor.Dimensions[3]}");
            using var results = _inferenceSession.Run(inputs);
            var output = results.FirstOrDefault(x => modelOutputs.Contains(x.Name));

            if (output?.Value is not DenseTensor<T> outputTensor)
                throw new Exception("Inference output không hợp lệ.");

            return outputTensor;
        }
        finally {
            // Giải phóng bộ nhớ
            foreach (var mat in resizedImages) {
                if (mat != null && !mat.IsDisposed)
                    mat.Dispose();
            }
        }
    }

    public Label[] getLabels() => Labels;

    public virtual TPrediction[] Predict(Mat image) {
        throw new NotImplementedException();
    }

    public virtual TPrediction[] BatchPredict(List<Mat> images) {
        throw new NotImplementedException();
    }
}
