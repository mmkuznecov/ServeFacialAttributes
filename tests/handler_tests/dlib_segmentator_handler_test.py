import base64
import cv2
import numpy as np
from src.handlers.dlib_segmentator_handler.dlib_segmentator_handler import (
    DlibSegmentatorHandler,
)
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
        "model_dir": "models/dlib",
        "serialized_file": "weights/shape_predictor_81_face_landmarks.dat",
    }
]

IMAGE_PATHS = ["tests/test_images/bald.jpg", "tests/test_images/not_bald.jpg"]

parametrize_mock_context = create_parametrize_mock_context(MOCK_PARAMS)
parametrize_image_path = create_parametrize_image_path(IMAGE_PATHS)

handler_instance = create_handler_instance(DlibSegmentatorHandler)


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
def test_postprocess(handler_instance, inference_output, preprocessed_image):
    postprocessed_output = handler_instance.postprocess(inference_output)
    assert isinstance(postprocessed_output, list), "Output should be a list"
    assert len(postprocessed_output) > 0, "Output list should not be empty"
    assert (
        "segmentation_mask" in postprocessed_output[0]
    ), "Output should contain 'segmentation_mask'"
    assert isinstance(
        postprocessed_output[0]["segmentation_mask"], str
    ), "'segmentation_mask' should be a string"

    for output in postprocessed_output:
        base64_mask = output["segmentation_mask"]
        decoded_mask = base64.b64decode(base64_mask)
        mask_array = np.frombuffer(decoded_mask, dtype=np.uint8)
        mask = cv2.imdecode(mask_array, cv2.IMREAD_UNCHANGED)

        original_image = preprocessed_image[0]
        assert (
            mask.shape[:2] == original_image.shape[:2]
        ), "Size of the segmented mask should match the original image"

        unique_pixels = np.unique(mask)
        expected_pixels = [0, 20, 30, 40, 50, 60, 70, 80]
        assert set(unique_pixels) == set(
            expected_pixels
        ), "Segmented mask should contain all expected pixel types"

    print("Postprocessed data:", postprocessed_output)
