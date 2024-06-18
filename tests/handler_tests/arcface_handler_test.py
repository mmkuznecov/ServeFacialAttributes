import pytest
import os
from PIL import Image
from ..test_utils import load_image_as_request_input, mock_context
from src.handlers.arcface_handler.arcface_handler import ResnetArcfaceHandler

MOCK_PARAMS = [
    {
        "model_dir": "models/arcface",
        "serialized_file": "weights/resnet18_110_arcface.pth",
    }
]

parametrize_mock_context = pytest.mark.parametrize(
    "mock_context", MOCK_PARAMS, indirect=True
)


@pytest.fixture
def sample_image_path():
    return os.path.join("tests/test_images", "not_bald.jpg")


@pytest.fixture
def handler_instance(mock_context):
    handler = ResnetArcfaceHandler()
    handler.initialize(mock_context)
    return handler


@pytest.fixture
def req_input(sample_image_path):
    return load_image_as_request_input(sample_image_path)


@pytest.fixture
def preprocessed_image(handler_instance, req_input):
    return handler_instance.preprocess(req_input)


@pytest.fixture
def embeddings(handler_instance, preprocessed_image):
    return handler_instance.inference(preprocessed_image)


@parametrize_mock_context
def test_preprocess(handler_instance, req_input):
    preprocessed_image = handler_instance.preprocess(req_input)
    assert len(preprocessed_image) == 1
    assert isinstance(preprocessed_image[0], Image.Image)
    print("Preprocessed data:", preprocessed_image)


@parametrize_mock_context
def test_inference(handler_instance, preprocessed_image):
    embeddings = handler_instance.inference(preprocessed_image)
    assert len(embeddings) == 1
    assert len(embeddings[0]) == 512
    print("Inference result:", embeddings)


@parametrize_mock_context
def test_postprocess(handler_instance, embeddings):
    final_result = handler_instance.postprocess(embeddings)
    assert len(final_result) == 1
    assert isinstance(final_result[0], list)
    assert len(final_result[0]) == 512
    print("Postprocessed data:", final_result)
