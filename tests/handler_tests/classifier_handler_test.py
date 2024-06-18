import pytest
import os
from PIL import Image
from src.handlers.classifiers_handler.classifier_handler import (
    ResnetClassifierModelHandler,
)
from ..test_utils import load_image_as_request_input, mock_context

MOCK_PARAMS = [
    {"model_dir": "models/baldness", "serialized_file": "weights/bald_weights.pth"},
    {"model_dir": "models/beard", "serialized_file": "weights/beard_weights.pth"},
    {"model_dir": "models/emotions", "serialized_file": "weights/emotion_model.pth"},
    {"model_dir": "models/gender", "serialized_file": "weights/gender_model.pth"},
    {"model_dir": "models/glasses", "serialized_file": "weights/glasses_weights.pth"},
    {"model_dir": "models/happiness", "serialized_file": "weights/happy_model.pth"},
    {"model_dir": "models/race", "serialized_file": "weights/race_model.pth"},
]

parametrize_mock_context = pytest.mark.parametrize(
    "mock_context", MOCK_PARAMS, indirect=True
)


@pytest.fixture
def sample_image_path():
    return os.path.join("tests/test_images", "not_bald.jpg")


@pytest.fixture
def handler_instance(mock_context):
    handler = ResnetClassifierModelHandler()
    handler.initialize(mock_context)
    return handler


@pytest.fixture
def req_input(sample_image_path):
    return load_image_as_request_input(sample_image_path)


@pytest.fixture
def preprocessed_image(handler_instance, req_input):
    return handler_instance.preprocess(req_input)


@pytest.fixture
def inference_output(handler_instance, preprocessed_image):
    return handler_instance.inference(preprocessed_image)


@parametrize_mock_context
def test_preprocess(handler_instance, req_input):
    preprocessed_image = handler_instance.preprocess(req_input)
    assert len(preprocessed_image) == 1
    assert isinstance(preprocessed_image[0], Image.Image)
    print("Preprocessed data:", preprocessed_image)


@parametrize_mock_context
def test_inference(handler_instance, preprocessed_image):
    result = handler_instance.inference(preprocessed_image)
    assert isinstance(result, list), "Inference result should be a list"
    print("Inference result:", result)


@parametrize_mock_context
def test_postprocess(handler_instance, inference_output):
    final_result = handler_instance.postprocess(inference_output)
    assert isinstance(final_result, list), "Final result should be a list"
    print("Postprocessed data:", final_result)
