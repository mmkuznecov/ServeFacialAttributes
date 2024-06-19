import pytest
import torch
import numpy as np
from PIL import Image
from src.handlers.deeplab_segmentator_handler.deeplab_segmentator import (
    FaceSegmentationPredictor,
)


@pytest.fixture
def model_path():
    return "models/deeplabv3_face/weights/deeplabv3plus_celebamask.pth"


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def predictor(model_path, device):
    return FaceSegmentationPredictor(model_path=model_path, device=device)


@pytest.fixture
def sample_image():
    # Create a dummy image for testing
    width, height = 256, 256
    image = Image.new("RGB", (width, height), color="white")
    return image


def test_predictor_initialization(predictor):
    assert predictor.model is not None, "Model should be initialized"
    assert predictor.device in [
        "cpu",
        "cuda",
    ], "Device should be either 'cpu' or 'cuda'"
    assert predictor.transform is not None, "Transform should be initialized"
    assert (
        predictor.CLASS_COLOR_MAP is not None
    ), "CLASS_COLOR_MAP should be initialized"


def test_predict_single_image(predictor, sample_image):
    result = predictor.predict(sample_image)
    assert isinstance(result, Image.Image), "Result should be a PIL Image"
    assert result.size == sample_image.size, "Result size should match input image size"


def test_predict_image_content(predictor, sample_image):
    result = predictor.predict(sample_image)
    result_array = np.array(result)
    assert result_array.shape == (
        sample_image.size[1],
        sample_image.size[0],
        3,
    ), "Result shape should match input image shape"
    unique_colors = set(tuple(color) for m2d in result_array for color in m2d)
    expected_colors = set(predictor.CLASS_COLOR_MAP.values())
    assert unique_colors.issubset(
        expected_colors
    ), "Result should only contain colors from CLASS_COLOR_MAP"
