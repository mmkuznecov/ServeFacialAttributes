import pytest
import numpy as np
from src.handlers.skincolor_handler.skincolor_handler import SkinColorHandler
from ..test_utils import load_image_as_request_input, mock_context

MOCK_PARAMS = [
    {
        "model_dir": "models/skincolor",
        "serialized_file": "weights/skincolor_weights.pth",
    }
]

IMAGE_PATHS = [
    (
        "tests/test_images/skincolor_samples/00000.png",
        "tests/test_images/skincolor_samples/00000_mask.png",
    ),
]

parametrize_mock_context = pytest.mark.parametrize(
    "mock_context", MOCK_PARAMS, indirect=True
)
parametrize_image_path = pytest.mark.parametrize("image_path, mask_path", IMAGE_PATHS)


@pytest.fixture
def handler_instance(mock_context):
    handler = SkinColorHandler()
    handler.initialize(mock_context)
    return handler


@pytest.fixture
def req_input(image_path, mask_path):
    image_request_input = load_image_as_request_input(image_path, encode_base64=True)
    mask_request_input = load_image_as_request_input(mask_path, encode_base64=True)
    return [
        {
            "body": {
                "image": image_request_input[0]["body"],
                "mask": mask_request_input[0]["body"],
            }
        }
    ]


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
    assert "image" in preprocessed_image[0]
    assert "mask" in preprocessed_image[0]
    assert isinstance(preprocessed_image[0]["image"], np.ndarray)
    assert isinstance(preprocessed_image[0]["mask"], np.ndarray)
    print("Preprocessed data:", preprocessed_image)


@parametrize_mock_context
@parametrize_image_path
def test_inference(handler_instance, preprocessed_image):
    inference_output = handler_instance.inference(preprocessed_image)
    assert isinstance(inference_output, list), "Inference output should be a list"
    assert len(inference_output) > 0, "Inference output list should not be empty"

    expected_keys = ["lum", "hue", "lum_std", "hue_std", "a_values", "b_values"]
    for key in expected_keys:
        assert key in inference_output[0], f"Inference output should contain '{key}'"

    print("Inference result:", inference_output)


@parametrize_mock_context
@parametrize_image_path
def test_postprocess(handler_instance, inference_output):
    postprocessed_output = handler_instance.postprocess(inference_output)
    assert isinstance(postprocessed_output, list), "Output should be a list"
    assert len(postprocessed_output) > 0, "Output list should not be empty"

    expected_keys = ["lum", "hue", "lum_std", "hue_std", "a_values", "b_values"]
    for key in expected_keys:
        assert key in postprocessed_output[0], f"Output should contain '{key}'"

    assert isinstance(postprocessed_output[0]["lum"], float), "'lum' should be a float"
    assert isinstance(postprocessed_output[0]["hue"], float), "'hue' should be a float"
    assert isinstance(
        postprocessed_output[0]["lum_std"], float
    ), "'lum_std' should be a float"
    assert isinstance(
        postprocessed_output[0]["hue_std"], float
    ), "'hue_std' should be a float"
    assert isinstance(
        postprocessed_output[0]["a_values"], list
    ), "'a_values' should be a list"
    assert isinstance(
        postprocessed_output[0]["b_values"], list
    ), "'b_values' should be a list"

    assert (
        0 <= postprocessed_output[0]["lum"] <= 100
    ), "'lum' should be in the range [0, 100]"
    assert (
        0 <= postprocessed_output[0]["hue"] <= 90
    ), "'hue' should be in the range [0, 90]"

    lum_threshold = 60
    hue_threshold = 55
    skin_tone = "light" if postprocessed_output[0]["lum"] > lum_threshold else "dark"
    skin_hue = "red" if postprocessed_output[0]["hue"] < hue_threshold else "yellow"

    print("Postprocessed data:", postprocessed_output)
    print(f"Skin tone: {skin_tone}, Skin hue: {skin_hue}")