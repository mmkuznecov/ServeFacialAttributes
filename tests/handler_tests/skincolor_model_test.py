import pytest
import numpy as np
from skimage import data
from skimage.color import rgb2gray, rgb2lab
from src.handlers.skincolor_handler.skincolor_calculator import SkinColorPredictor


@pytest.fixture
def sample_image():
    # Using a sample image from skimage for testing
    return data.astronaut()


@pytest.fixture
def sample_mask(sample_image):
    # Creating a mask that selects the face area in the sample image
    gray_image = rgb2gray(sample_image)
    mask = np.zeros_like(gray_image, dtype=np.uint8)
    mask[80:180, 80:180] = 1  # A simple square mask over the face
    return mask


@pytest.fixture
def skin_color_predictor():
    return SkinColorPredictor()


def test_get_hue(skin_color_predictor):
    a_values = np.array([1, 0, -1])
    b_values = np.array([1, 1, 1])
    hues = skin_color_predictor.get_hue(a_values, b_values)
    assert isinstance(hues, np.ndarray), "Hue should be a numpy array"
    assert hues.shape == a_values.shape, "Hue should have the same shape as input arrays"


def test_mode_hist(skin_color_predictor):
    x = np.random.normal(size=1000)
    mode = skin_color_predictor.mode_hist(x)
    assert isinstance(mode, float), "Mode should be a float"


def test_clustering(skin_color_predictor, sample_image, sample_mask):
    img = sample_image[sample_mask == 1]
    labels, model = skin_color_predictor.clustering(img)
    assert len(labels) == img.shape[0], "Labels should match the number of skin pixels"
    assert hasattr(model, 'cluster_centers_'), "Model should have cluster centers"


def test_get_scalar_values(skin_color_predictor, sample_image, sample_mask):
    img = sample_image[sample_mask == 1]
    img_lab = rgb2lab(img)
    labels, _ = skin_color_predictor.clustering(img_lab[:, :2])
    scalar_values = skin_color_predictor.get_scalar_values(img_lab, labels)
    assert isinstance(scalar_values, dict), "Result should be a dictionary"
    for key in ["lum", "hue", "lum_std", "hue_std"]:
        assert key in scalar_values, f"Result should contain '{key}'"


def test_get_skin_values(skin_color_predictor, sample_image, sample_mask):
    skin_values = skin_color_predictor.get_skin_values(sample_image, sample_mask)
    assert isinstance(skin_values, dict), "Result should be a dictionary"
    for key in ["lum", "hue", "lum_std", "hue_std", "red", "green", "blue", "red_std", "green_std", "blue_std"]:
        assert key in skin_values, f"Result should contain '{key}'"


def test_predict(skin_color_predictor, sample_image, sample_mask):
    result = skin_color_predictor.predict(sample_image, sample_mask)
    assert isinstance(result, dict), "Result should be a dictionary"
    for key in ["lum", "hue", "lum_std", "hue_std", "a_values", "b_values"]:
        assert key in result, f"Result should contain '{key}'"
    assert isinstance(result["a_values"], np.ndarray), "'a_values' should be a numpy array"
    assert isinstance(result["b_values"], np.ndarray), "'b_values' should be a numpy array"

