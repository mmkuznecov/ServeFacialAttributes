import pytest
import torch
import numpy as np
from PIL import Image
from src.handlers.headpose_handler.sixdrepnet360 import HeadPoseEstimator, SixDRepNet360
import torchvision
import os

@pytest.fixture
def weights_path():
    return "models/headpose/weights/headpose_weights.pth"

@pytest.fixture
def estimator(weights_path):
    return HeadPoseEstimator(weights_url=weights_path)

@pytest.fixture
def sample_image_path():
    return os.path.join("tests/test_images", "not_bald.jpg")

@pytest.fixture
def sample_image(sample_image_path):
    return Image.open(sample_image_path)

def test_sixdrepnet360_initialization():
    model = SixDRepNet360(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3])
    assert isinstance(model, SixDRepNet360), "Model should be an instance of SixDRepNet360"
    assert isinstance(model.conv1, torch.nn.Conv2d), "conv1 should be an instance of Conv2d"
    assert isinstance(model.linear_reg, torch.nn.Linear), "linear_reg should be an instance of Linear"

def test_headpose_estimator_initialization(estimator):
    assert isinstance(estimator.model, SixDRepNet360), "Model should be an instance of SixDRepNet360"
    assert estimator.device in ["cuda", "cpu"], "Device should be either 'cuda' or 'cpu'"
    assert estimator.transform is not None, "Transform should be initialized"

def test_load_weights(estimator, weights_path):
    estimator.load_weights(weights_path)
    assert estimator.model is not None, "Model should be initialized after loading weights"

def test_process_image(estimator, sample_image):
    image_tensor = estimator.process_image(sample_image)
    assert isinstance(image_tensor, torch.Tensor), "Transformed image should be a tensor"
    assert image_tensor.shape == (3, 224, 224), "Transformed image shape should be (3, 224, 224)"

def test_predict_single_image(estimator, sample_image):
    prediction = estimator.predict(sample_image)
    assert isinstance(prediction, np.ndarray), "Prediction should be a numpy array"
    assert prediction.shape == (1, 3), f"Prediction shape should be (1, 3)"

def test_predict_batch_images(estimator, sample_image):
    images = [sample_image for _ in range(4)]  # Create a batch of 4 images
    predictions = estimator.predict(images)
    assert isinstance(predictions, np.ndarray), "Predictions should be a numpy array"
    assert predictions.shape == (4, 3), f"Predictions shape should be (4, 3)"

def test_predict_different_images(estimator):
    image1 = Image.new('RGB', (256, 256), color='red')
    image2 = Image.new('RGB', (256, 256), color='blue')
    images = [image1, image2]  # Batch with different images
    predictions = estimator.predict(images)
    assert isinstance(predictions, np.ndarray), "Predictions should be a numpy array"
    assert predictions.shape == (2, 3), f"Predictions shape should be (2, 3)"
