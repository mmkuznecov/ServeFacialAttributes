import numpy as np
from PIL import Image
from src.handlers.headpose_handler.headposehandler import SixDRepNet360Handler
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
        "model_dir": "models/headpose",
        "serialized_file": "weights/headpose_weights.pth",
    }
]

IMAGE_PATHS = ["tests/test_images/not_bald.jpg"]

parametrize_mock_context = create_parametrize_mock_context(MOCK_PARAMS)
parametrize_image_path = create_parametrize_image_path(IMAGE_PATHS)

handler_instance = create_handler_instance(SixDRepNet360Handler)


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
    inference_output = handler_instance.inference(preprocessed_image)
    assert isinstance(
        inference_output, (list, np.ndarray)
    ), "Inference output should be a list or numpy array"
    assert len(inference_output) > 0, "Inference output list should not be empty"
    print("Inference result:", inference_output)


@parametrize_mock_context
@parametrize_image_path
def test_postprocess(handler_instance, inference_output):
    postprocessed_output = handler_instance.postprocess(inference_output)
    assert isinstance(postprocessed_output, list), "Output should be a list"
    assert len(postprocessed_output) > 0, "Output list should not be empty"
    for output in postprocessed_output:
        assert all(
            key in output for key in ["yaw", "pitch", "roll"]
        ), "Output missing expected keys"
        assert all(
            isinstance(output[key], float) for key in ["yaw", "pitch", "roll"]
        ), "Values should be float"
    print("Postprocessed data:", postprocessed_output)
