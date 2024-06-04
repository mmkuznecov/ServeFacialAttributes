import pytest
from src.handlers.face_detection_handler.face_detection_handler import (
    YOLOv8FaceDetectionHandler,
)
from ..test_utils import load_image_as_request_input, mock_context


@pytest.mark.parametrize(
    "mock_context",
    [
        {
            "model_dir": "models/face_detection",
            "serialized_file": "weights/yolov8n-face.pt",
        }
    ],
    indirect=True,
)
def test_yolov8_face_detection_handler(mock_context):
    image_path = "tests/test_images/not_bald.jpg"
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
