import pytest
import io
import base64
import cv2
import numpy as np
from PIL import Image

from src.handlers.deeplab_segmentator_handler.deeplab_segmentator_handler import (
    FaceSegmentationHandler
)
from src.utils.test_utils import load_image_as_request_input, mock_context


@pytest.mark.parametrize(
    "mock_context",
    [
        {
            "model_dir": "models/deeplabv3_face",
            "serialized_file": "weights/deeplabv3plus_celebamask.pth",
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "image_path", ["test_images/bald.jpg", "test_images/not_bald.jpg"]
)
def test_face_segmentation_handler(mock_context, image_path):
    request_input = load_image_as_request_input(image_path)
    # Initialize handler instance
    handler = FaceSegmentationHandler()
    handler.initialize(mock_context)

    preprocessed_input = handler.preprocess(request_input)
    inference_output = handler.inference(preprocessed_input)
    postprocessed_output = handler.postprocess(inference_output)

    # Assertions to validate the output
    assert isinstance(postprocessed_output, list), "Output should be a list"
    assert len(postprocessed_output) > 0, "Output list should not be empty"
    assert (
        "segmentation_mask" in postprocessed_output[0]
    ), "Output should contain 'segmentation_mask'"
    assert isinstance(
        postprocessed_output[0]["segmentation_mask"], str
    ), "'segmentation_mask' should be a string"

    # Decode the base64-encoded segmentation mask
    for output in postprocessed_output:
        base64_mask = output["segmentation_mask"]
        decoded_mask = base64.b64decode(base64_mask)
        mask_array = np.frombuffer(decoded_mask, dtype=np.uint8)
        mask = cv2.imdecode(mask_array, cv2.IMREAD_UNCHANGED)

        # Check if the size of the original image and segmented mask are equal
        original_image = preprocessed_input[0]
        assert (
            mask.shape[:2] == original_image.size[::-1]
        ), "Size of the segmented mask should match the original image"

        # Check if all expected colors are present in the segmented mask
        unique_colors = set(tuple(color) for m2d in mask for color in m2d)
        expected_colors = set(handler.model.CLASS_COLOR_MAP.values())
        assert (
            unique_colors.issubset(expected_colors)
        ), "Segmented mask should only contain expected colors"

    print(f"Test passed for image: {image_path}")