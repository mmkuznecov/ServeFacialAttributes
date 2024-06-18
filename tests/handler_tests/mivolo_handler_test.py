import pytest
import torch
from src.handlers.mivolo_handler.mivolo_handler import MiVOLOHandler
from ..test_utils import load_image_as_request_input, mock_context

MOCK_PARAMS = [
    {
        "model_dir": "models/age",
        "serialized_file": "weights/model_age_utk_4.23.pth.tar",
        "gpu_id": 0 if torch.cuda.is_available() else None,
    }
]

IMAGE_PATHS = ["tests/test_images/bald.jpg", "tests/test_images/not_bald.jpg"]

parametrize_mock_context = pytest.mark.parametrize(
    "mock_context", MOCK_PARAMS, indirect=True
)
parametrize_image_path = pytest.mark.parametrize("image_path", IMAGE_PATHS)


@pytest.fixture
def handler_instance(mock_context):
    handler = MiVOLOHandler()
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
    assert preprocessed_image is not None
    assert isinstance(preprocessed_image, torch.Tensor)
    print("Preprocessed data:", preprocessed_image)


@parametrize_mock_context
@parametrize_image_path
def test_inference(handler_instance, preprocessed_image):
    inference_output = handler_instance.inference(preprocessed_image)
    assert isinstance(
        inference_output, torch.Tensor
    ), "Inference output should be a tensor"
    assert len(inference_output) > 0, "Inference output should not be empty"
    print("Inference result:", inference_output)


@parametrize_mock_context
@parametrize_image_path
def test_postprocess(handler_instance, inference_output):
    postprocessed_output = handler_instance.postprocess(inference_output)
    assert isinstance(postprocessed_output, list), "Output should be a list"
    assert len(postprocessed_output) == 1, "Output list should contain one item"
    output = postprocessed_output[0]
    assert "age" in output, "Output should contain 'age'"
    assert isinstance(output["age"], float), "'age' should be a float"
    assert 0 <= output["age"] <= 100, "'age' should be between 0 and 100"
    print("Postprocessed data:", postprocessed_output)
