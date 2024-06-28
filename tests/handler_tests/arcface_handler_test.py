import pytest
from PIL import Image
from src.handlers.arcface_handler.arcface_handler import ResnetArcfaceHandler
from ..test_utils import mock_context
from ..fixture_utils import (
    create_handler_instance,
    req_input,
    preprocessed_image,
    create_parametrize_mock_context,
    create_parametrize_image_path,
)

MOCK_PARAMS = [
    {
        "model_dir": "models/arcface",
        "serialized_file": "weights/resnet18_110_arcface.pth",
    }
]

IMAGE_PATHS = ["tests/test_images/not_bald.jpg"]

parametrize_mock_context = create_parametrize_mock_context(MOCK_PARAMS)
parametrize_image_path = create_parametrize_image_path(IMAGE_PATHS)

handler_instance = create_handler_instance(ResnetArcfaceHandler)


@pytest.fixture
def embeddings(handler_instance, preprocessed_image):
    return handler_instance.inference(preprocessed_image)


@parametrize_mock_context
@parametrize_image_path
def test_preprocess(handler_instance, req_input):
    preprocessed_image = handler_instance.preprocess(req_input)
    assert len(preprocessed_image) == 1
    assert isinstance(preprocessed_image[0], Image.Image)
    print("Preprocessed data:", preprocessed_image)


@parametrize_mock_context
@parametrize_image_path
def test_inference(handler_instance, preprocessed_image):
    embeddings = handler_instance.inference(preprocessed_image)
    assert len(embeddings) == 1
    assert len(embeddings[0]) == 512
    print("Inference result:", embeddings)


@parametrize_mock_context
@parametrize_image_path
def test_postprocess(handler_instance, embeddings):
    final_result = handler_instance.postprocess(embeddings)
    assert len(final_result) == 1
    assert isinstance(final_result[0], list)
    assert len(final_result[0]) == 512
    print("Postprocessed data:", final_result)
