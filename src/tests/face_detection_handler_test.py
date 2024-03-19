import pytest
from unittest.mock import Mock
from src.handlers.face_detection_handler.face_detection_handler import (
    YOLOv8FaceDetectionHandler,
)


@pytest.fixture
def mock_context():
    context = Mock()
    context.system_properties = {"model_dir": "models/face_detection", "gpu_id": "0"}
    context.manifest = {"model": {"serializedFile": "weights/yolov8n-face.pt"}}
    return context


def load_image_as_request_input(image_path):
    with open(image_path, "rb") as img_file:
        image_bytes = img_file.read()
    return [{"body": image_bytes}]


def test_yolov8_face_detection_handler(mock_context):
    image_path = "test_images/not_bald.jpg"
    request_input = load_image_as_request_input(image_path)

    # Initialize handler instance and methods
    handler = YOLOv8FaceDetectionHandler()
    handler.initialize(mock_context)
    preprocessed_input = handler.preprocess(request_input)
    inference_output = handler.inference(preprocessed_input)
    postprocessed_output = handler.postprocess(inference_output)

    # Assertions to validate the output
    assert isinstance(postprocessed_output, list), "Output should be a list"
    assert len(postprocessed_output) > 0, "Output list should not be empty"
    assert "boxes" in postprocessed_output[0], "Output should contain 'boxes'"
    assert isinstance(
        postprocessed_output[0]["boxes"], list
    ), "'boxes' should be a list"
    assert len(postprocessed_output[0]["boxes"]) > 0, "'boxes' list should not be empty"
    for box in postprocessed_output[0]["boxes"]:
        assert len(box) == 4, "Each box should have 4 values (x, y, width, height)"
        assert all(
            isinstance(value, float) for value in box
        ), "Box values should be floats"

    print("Test passed: YOLOv8 face detection handler output format is correct.")
