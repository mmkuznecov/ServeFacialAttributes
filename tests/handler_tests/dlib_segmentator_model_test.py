import pytest
import numpy as np
import cv2
from src.handlers.dlib_segmentator_handler.dlib_segmentator import DlibSegmentator, SEGMENTATION_COLOR_MAPPING
import os

@pytest.fixture
def predictor_path():
    return "models/dlib/weights/shape_predictor_81_face_landmarks.dat"

@pytest.fixture
def segmentator(predictor_path):
    return DlibSegmentator(predictor_path=predictor_path)

@pytest.fixture
def sample_image_path():
    return os.path.join("tests/test_images", "not_bald.jpg")

@pytest.fixture
def sample_image(sample_image_path):
    image = cv2.imread(sample_image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for consistency with dlib

def test_segmentator_initialization(segmentator):
    assert segmentator.detector is not None, "Detector should be initialized"
    assert segmentator.predictor is not None, "Predictor should be initialized"

def test_segment_face(segmentator, sample_image):
    mask = segmentator.segment_face(sample_image)
    assert isinstance(mask, np.ndarray), "Mask should be a numpy array"
    assert mask.shape == sample_image.shape[:2], "Mask shape should match input image shape"
    assert mask.dtype == np.uint8, "Mask dtype should be uint8"
    assert np.any(mask != 0), "Mask should contain non-zero values for the face regions"
    assert len(np.unique(mask)) > 1, "Mask should contain more than one unique value"

def test_segment_face_content(segmentator, sample_image):
    mask = segmentator.segment_face(sample_image)
    unique_colors = np.unique(mask)
    expected_colors = list(SEGMENTATION_COLOR_MAPPING.values())
    for color in unique_colors:
        assert color in expected_colors or color == 0, f"Unexpected color value {color} in mask"

def test_no_faces(segmentator):
    # Create a dummy image with no face-like structures for testing
    width, height = 256, 256
    image = np.zeros((height, width, 3), dtype=np.uint8)
    mask = segmentator.segment_face(image)
    assert isinstance(mask, np.ndarray), "Mask should be a numpy array"
    assert mask.shape == image.shape[:2], "Mask shape should match input image shape"
    assert mask.dtype == np.uint8, "Mask dtype should be uint8"
    assert np.all(mask == 0), "Mask should contain only zero values as there are no faces"
