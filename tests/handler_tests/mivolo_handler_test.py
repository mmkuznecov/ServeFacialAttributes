import torch
from src.handlers.mivolo_handler.mivolo_handler import MiVOLOHandler
from ..test_utils import mock_context
from ..fixture_utils import (
    create_handler_instance,
    req_input,
    preprocessed_image,
    inference_output,
    create_parametrize_mock_context,
    create_parametrize_image_path,
)

MOCK_PARAMS = [
    {
        "model_dir": "models/age",
        "serialized_file": "weights/model_age_utk_4.23.pth.tar",
        "gpu_id": 0 if torch.cuda.is_available() else None,
    }
]

IMAGE_PATHS = ["tests/test_images/bald.jpg", "tests/test_images/not_bald.jpg"]

parametrize_mock_context = create_parametrize_mock_context(MOCK_PARAMS)
parametrize_image_path = create_parametrize_image_path(IMAGE_PATHS)

handler_instance = create_handler_instance(MiVOLOHandler)


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
