import pytest
import numpy as np
from src.handlers.skincolor_handler.skincolor_handler import SkinColorHandler
from src.utils.test_utils import load_image_as_request_input, mock_context

@pytest.mark.parametrize(
    "image_path, mask_path",
    [
        ("test_images/skincolor_samples/00000.png",
         "test_images/skincolor_samples/00000_mask.png"),
    ],
)
def test_skin_color_handler(mock_context, image_path, mask_path):
    # Load image and mask as request input
    image_request_input = load_image_as_request_input(image_path)
    mask_request_input = load_image_as_request_input(mask_path)
    request_input = [
        {"image": image_request_input[0]["body"], "mask": mask_request_input[0]["body"]}
    ]

    # Initialize handler instance
    handler = SkinColorHandler()
    handler.initialize(mock_context)

    preprocessed_input = handler.preprocess(request_input)
    inference_output = handler.inference(preprocessed_input)
    postprocessed_output = handler.postprocess(inference_output)

    # Assertions to validate the output
    assert isinstance(postprocessed_output, list), "Output should be a list"
    assert len(postprocessed_output) > 0, "Output list should not be empty"

    expected_keys = ["lum", "hue", "lum_std", "hue_std", "a_values", "b_values"]
    for key in expected_keys:
        assert key in postprocessed_output[0], f"Output should contain '{key}'"

    assert isinstance(postprocessed_output[0]["lum"], float), "'lum' should be a float"
    assert isinstance(postprocessed_output[0]["hue"], float), "'hue' should be a float"
    assert isinstance(postprocessed_output[0]["lum_std"], float), "'lum_std' should be a float"
    assert isinstance(postprocessed_output[0]["hue_std"], float), "'hue_std' should be a float"
    assert isinstance(postprocessed_output[0]["a_values"], list), "'a_values' should be a list"
    assert isinstance(postprocessed_output[0]["b_values"], list), "'b_values' should be a list"
    # Verify value ranges based on the information from the supplementary material
    assert 0 <= postprocessed_output[0]["lum"] <= 100, "'lum' should be in the range [0, 100]"
    assert 0 <= postprocessed_output[0]["hue"] <= 90, "'hue' should be in the range [0, 90]"

    # Verify skin tone and hue categorization
    lum_threshold = 60
    hue_threshold = 55
    if postprocessed_output[0]["lum"] > lum_threshold:
        skin_tone = "light"
    else:
        skin_tone = "dark"

    if postprocessed_output[0]["hue"] < hue_threshold:
        skin_hue = "red"
    else:
        skin_hue = "yellow"

    print(
        f"Test passed for image: {image_path}, mask: {mask_path} with output: {postprocessed_output[0]}"
    )
    print(f"Skin tone: {skin_tone}, Skin hue: {skin_hue}")
