using System.Runtime.InteropServices;
using ByteTrack;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace OnnxZoo.Extensions;
public static class Utils {
    public static int[] Xywh2xyxy(float[] source) {
        var result = new int[4];

        result[0] = (int)(source[0] - source[2] / 2f);
        result[1] = (int)(source[1] - source[3] / 2f);
        result[2] = (int)(source[0] + source[2] / 2f);
        result[3] = (int)(source[1] + source[3] / 2f);

        return result;
    }

    public static int[] Xyxy2xywh(int[] source) {
        var result = new int[4];
        //result[0] = (source[0] + source[2]) / 2;
        //result[1] = (source[1] + source[3]) / 2;
        result[0] = source[0];
        result[1] = source[1];
        result[2] = source[2] - source[0];
        result[3] = source[3] - source[1];
        return result;
    }

    public static Mat Resize(Mat img, int width, int height) {
        Mat resized = new Mat();
        Cv2.Resize(img, resized, new Size(width, height));
        return resized;
    }

    public static DenseTensor<float> ExtractPixels(Mat image) {
        int channels = image.Channels();
        int height = image.Rows;
        int width = image.Cols;

        // Tạo tensor
        var tensor = new DenseTensor<float>(new[] { 1, channels, height, width });

        byte[] imageData = new byte[image.Total() * image.ElemSize()];
        Marshal.Copy(image.Data, imageData, 0, imageData.Length);

        int stride = width * channels;
        Parallel.For(0, height, h => {
            int rowOffset = h * stride;
            for (int w = 0; w < width; w++) {
                int index = rowOffset + w * channels;
                for (int c = 0; c < channels; c++) {
                    tensor[0, c, h, w] = imageData[index + c] / 255.0f;
                }
            }
        });

        return tensor;
    }

    public static Half[] Float32ToFloat16(float[] float32Array)
    {
        return float32Array.Select(f => (Half)f).ToArray();
    }

    public static float[] ComputeHogFeatures(Mat image) {
        Mat gray = new Mat();
        //Cv2.CvtColor(image, gray, ColorConversionCodes.BGR2GRAY);
        gray = image;

        Mat dx = new Mat();
        Mat dy = new Mat();
        Cv2.Sobel(gray, dx, MatType.CV_32F, 1, 0, 3);
        Cv2.Sobel(gray, dy, MatType.CV_32F, 0, 1, 3);

        Mat magnitude = new Mat();
        Mat orientation = new Mat();
        Cv2.Magnitude(dx, dy, magnitude);
        Cv2.Phase(dx, dy, orientation, angleInDegrees: true);

        for (int y = 0; y < orientation.Rows; y++)
            for (int x = 0; x < orientation.Cols; x++)
                orientation.At<float>(y, x) = orientation.At<float>(y, x) % 180f;

        int cellSize = 8;
        int numBins = 9;
        float binWidth = 180f / numBins;

        int numCellsX = image.Cols / cellSize;
        int numCellsY = image.Rows / cellSize;
        int totalCells = numCellsX * numCellsY;

        float[] featureVector = new float[totalCells * numBins];

        int featureIndex = 0;
        for (int cellY = 0; cellY < numCellsY; cellY++) {
            for (int cellX = 0; cellX < numCellsX; cellX++) {
                float[] hist = new float[numBins];

                for (int y = cellY * cellSize; y < (cellY + 1) * cellSize; y++) {
                    for (int x = cellX * cellSize; x < (cellX + 1) * cellSize; x++) {
                        float mag = magnitude.At<float>(y, x);
                        float ori = orientation.At<float>(y, x);

                        float index = ori / binWidth;
                        int left = (int)Math.Floor(index);
                        float frac = index - left;

                        if (left < numBins - 1) {
                            hist[left] += (1 - frac) * mag;
                            hist[left + 1] += frac * mag;
                        }
                        else {
                            hist[numBins - 1] += mag;
                        }
                    }
                }

                for (int b = 0; b < numBins; b++)
                    featureVector[featureIndex++] = hist[b];
            }
        }

        return featureVector;
    }

    public static float Clamp(float value, float min, float max) {
        return (value < min) ? min : (value > max) ? max : value;
    }

    public static Rect ToRect(this RectBox rect) => new ((int) rect.X, (int) rect.Y, (int) rect.Width, (int) rect.Height);

    public static Rect2f ToRect2f(this RectBox rect) => new(rect.X, rect.Y, rect.Width, rect.Height);
}
