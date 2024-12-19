from divevision.src.datasets.lsui_dataset import LSUIDataset
from divevision.src.models.u_shape_model import UShapeModelWrapper


def simple_test_routine() -> None:
    model = UShapeModelWrapper(training=False)
    dataset = LSUIDataset(transform=model.preprocessing)

    input, label = dataset[0]
    output = model.forward(input)

    input_img = model.postprocessing(input)
    output_img = model.postprocessing(output)
    label_img = model.postprocessing(label)

    print(f"Input shape: {input.shape}")
    print(f"Output shape: {output.shape}")
    input_img.save("input.jpg")
    output_img.save("output.jpg")
    label_img.save("label.jpg")


if __name__ == "__main__":
    simple_test_routine()
