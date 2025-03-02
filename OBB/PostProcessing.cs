using System.Collections.Immutable;
using System.Diagnostics;
using System.Runtime.InteropServices;
using OnnxZoo.Extensions;
using OpenCvSharp;
using static OpenCvSharp.FileStorage;

namespace OnnxZoo.OBB;

public static class PostProcessing {
    public static Prediction[] Suppress(Prediction[] predictions, float overlapThresh) {
        if (predictions.Length == 0) return Array.Empty<Prediction>();

        // Sắp xếp theo độ tin cậy giảm dần
        Array.Sort(predictions, (a, b) => b.Score.CompareTo(a.Score));

        bool[] suppressed = new bool[predictions.Length];
        List<int> keep = new List<int>(predictions.Length);

        // Tính toán trước các thông số cần thiết
        RotatedRect[] boxes = predictions.Select(p => p.Box).ToArray();
        float[] areas = boxes.Select(b => b.Size.Width * b.Size.Height).ToArray();

        for (int i = 0; i < predictions.Length; i++) {
            if (suppressed[i]) continue;

            keep.Add(i);
            var currentBox = boxes[i];

            // Parallel xử lý các box tiếp theo
            Parallel.For(i + 1, predictions.Length, j => {
                if (suppressed[j]) return;

                if (!FastBoundingCheck(currentBox, boxes[j])) return;

                double iou = CalculateRotatedIoU(currentBox, boxes[j]);
                if (iou > overlapThresh) {
                    suppressed[j] = true;
                }
            });
        }

        return keep.Select(index => predictions[index]).ToArray();
    }
    private static bool FastBoundingCheck(RotatedRect a, RotatedRect b) {
        float dx = Math.Abs(a.Center.X - b.Center.X);
        float dy = Math.Abs(a.Center.Y - b.Center.Y);
        float maxDistance = Math.Max(a.Size.Width, a.Size.Height) +
                          Math.Max(b.Size.Width, b.Size.Height);
        return dx <= maxDistance && dy <= maxDistance;
    }
    private static double CalculateRotatedIoU(RotatedRect box1, RotatedRect box2) {
        Point2f[] poly1 = box1.Points();
        Point2f[] poly2 = box2.Points();

        Point2f[] intersection;
        float intersectArea = Cv2.IntersectConvexConvex(poly1, poly2, out intersection);
        if (intersectArea <= 0)
            return 0.0;

        float area1 = box1.Size.Width * box1.Size.Height;
        float area2 = box2.Size.Width * box2.Size.Height;

        float unionArea = area1 + area2 - intersectArea;
        return intersectArea / unionArea;
    }

    //private static double CalculateRotatedIoU(RotatedRect box1, RotatedRect box2) {
    //    // Sử dụng native code cho tính toán hình học phức tạp
    //    return NativeMethods.CalculateRotatedIoU(box1, box2);
    //}

}

//internal static class NativeMethods {
//    [DllImport("opencv_world455", CallingConvention = CallingConvention.Cdecl)]
//    public static extern double CalculateRotatedIoU(RotatedRect rect1, RotatedRect rect2);
//}

