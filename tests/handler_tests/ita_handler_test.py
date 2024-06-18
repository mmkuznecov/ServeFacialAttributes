import pytest
import numpy as np
from src.handlers.ita_handler.ita_handler import ITAHandler
from ..test_utils import load_image_as_request_input, mock_context

MOCK_PARAMS = [
    {
        "model_dir": "models/dlib",
        "serialized_file": "weights/shape_predictor_81_face_landmarks.dat",
    }
]

IMAGE_PATHS = ["tests/test_images/bald.jpg", "tests/test_images/not_bald.jpg"]

parametrize_mock_context = pytest.mark.parametrize(
    "mock_context", MOCK_PARAMS, indirect=True
)
parametrize_image_path = pytest.mark.parametrize("image_path", IMAGE_PATHS)


@pytest.fixture
def handler_instance(mock_context):
    handler = ITAHandler()
    handler.initialize(mock_context)
    return handler


@pytest.fixture
def req_input(image_path):
    return load_image_as_request_input(image_path)


@pytest.fixture
def preprocessed_image(handler_instance, req_input):
    return handler_instance.preprocess(req_input)


@pytest.fixture
def inference_output(handler_instance, preprocessed_image):
    return handler_instance.inference(preprocessed_image)


@parametrize_mock_context
@parametrize_image_path
def test_preprocess(handler_instance, req_input):
    preprocessed_image = handler_instance.preprocess(req_input)
    assert len(preprocessed_image) == 1
    assert isinstance(preprocessed_image[0], np.ndarray)
    print("Preprocessed data:", preprocessed_image)


@parametrize_mock_context
@parametrize_image_path
def test_inference(handler_instance, preprocessed_image):
    inference_output = handler_instance.inference(preprocessed_image)
    assert isinstance(inference_output, list), "Inference output should be a list"
    assert len(inference_output) > 0, "Inference output list should not be empty"
    print("Inference result:", inference_output)


@parametrize_mock_context
@parametrize_image_path
def test_postprocess(handler_instance, inference_output):
    postprocessed_output = handler_instance.postprocess(inference_output)
    assert isinstance(postprocessed_output, list), "Output should be a list"
    assert len(postprocessed_output) > 0, "Output list should not be empty"
    assert "ita_value" in postprocessed_output[0], "Output should contain 'ita_value'"
    assert isinstance(
        postprocessed_output[0]["ita_value"], (list, float)
    ), "'ita_value' should be a list or float"

    for output in postprocessed_output:
        ita_value = (
            output["ita_value"][0]
            if isinstance(output["ita_value"], list)
            else output["ita_value"]
        )
        assert isinstance(ita_value, float), "ITA value should be a float"
    print("Postprocessed data:", postprocessed_output)
