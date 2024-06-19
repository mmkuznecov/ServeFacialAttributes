import pytest
import numpy as np
import cv2
from src.handlers.ita_handler.ita_calculator import (
    ITACalculator,
    extract_non_black,
    calc_ita,
)
import os


@pytest.fixture
def predictor_path():
    return "models/dlib/weights/shape_predictor_81_face_landmarks.dat"


@pytest.fixture
def ita_calculator(predictor_path):
    return ITACalculator(predictor_path=predictor_path)


@pytest.fixture
def sample_image_path():
    return os.path.join("tests/test_images", "not_bald.jpg")


@pytest.fixture
def sample_image(sample_image_path):
    return cv2.imread(sample_image_path)


@pytest.fixture
def no_face_image():
    # Create a dummy image with no face-like structures for testing
    width, height = 256, 256
    return np.zeros((height, width, 3), dtype=np.uint8)


def test_ita_calculator_initialization(ita_calculator):
    assert ita_calculator.detector is not None, "Detector should be initialized"
    assert ita_calculator.predictor is not None, "Predictor should be initialized"


def test_extract_non_black(sample_image):
    non_black_pixels = extract_non_black(sample_image)
    assert isinstance(
        non_black_pixels, np.ndarray
    ), "Extracted non-black pixels should be a numpy array"
    assert (
        non_black_pixels.shape[2] == 3
    ), "Extracted pixels should have 3 color channels"


def test_calc_ita(sample_image):
    non_black_pixels = extract_non_black(sample_image)
    ita_value = calc_ita(non_black_pixels)
    assert isinstance(ita_value, float), "ITA value should be a float"
    assert -90 <= ita_value <= 90, "ITA value should be between -90 and 90 degrees"


def test_calculate_ita(ita_calculator, sample_image):
    ita_value = ita_calculator.calculate_ita(sample_image)
    assert ita_value is not None, "ITA value should not be None when a face is detected"
    assert isinstance(ita_value, float), "ITA value should be a float"
    assert -90 <= ita_value <= 90, "ITA value should be between -90 and 90 degrees"


def test_calculate_ita_no_face(ita_calculator, no_face_image):
    ita_value = ita_calculator.calculate_ita(no_face_image)
    assert ita_value is None, "ITA value should be None when no face is detected"
