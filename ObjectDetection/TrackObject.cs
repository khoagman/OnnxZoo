using ByteTrack;

namespace OnnxZoo.ObjectDetection; 
public class TrackObject : IObject {
    readonly RectBox _box;
    readonly int _label;
    readonly float _prob;
    readonly string _name;

    public RectBox RectBox => _box;

    public int Label => _label;

    public float Prob => _prob;

    public string Name => _name;

    public TrackObject(float[] xywh, int label, float prob, string name) {
        // create a new RectBox object
        _box = new RectBox(xywh[0], xywh[1], xywh[2], xywh[3]);

        _label = label;
        _prob = prob;
        _name = name;
    }

    public Track ToTrack() => new Track(_box, _prob, ("name", _name), ("label", _label));
}
