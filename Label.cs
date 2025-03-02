namespace OnnxZoo; 
public class Label {
    public int Id { get; init; }

    public string? Name { get; init; }

    public LabelType Type { get; init; } = LabelType.Generic;
}
