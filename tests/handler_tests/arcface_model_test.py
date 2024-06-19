import pytest
import torch
from PIL import Image
import numpy as np
from src.handlers.arcface_handler.arcface_model import ArcfaceModel, load_model

@pytest.fixture
def weights_path():
    return "models/arcface/weights/resnet18_110_arcface.pth"

@pytest.fixture
def arcface_model(weights_path):
    return ArcfaceModel(weights_path)

def test_load_model(weights_path):
    model = load_model(weights_path)
    assert isinstance(model, torch.nn.Module), "Model should be an instance of torch.nn.Module"

def test_transform(arcface_model):
    image = Image.new('RGB', (256, 256), color = 'red')
    transformed_image = arcface_model.transform(image)
    assert isinstance(transformed_image, torch.Tensor), "Transformed image should be a tensor"
    assert transformed_image.shape == (1, 128, 128), "Transformed image shape should be (1, 128, 128)"

def test_predict(arcface_model):
    image = Image.new('RGB', (256, 256), color = 'red')
    images = [image for _ in range(4)]  # Create a batch of 4 images
    predictions = arcface_model.predict(images)
    assert isinstance(predictions, np.ndarray), "Predictions should be a numpy array"
    assert predictions.shape == (4, 512), "Predictions shape should be (4, 512)"
    assert not np.any(np.isnan(predictions)), "Predictions should not contain NaN values"

def test_predict_single_image(arcface_model):
    image = Image.new('RGB', (256, 256), color = 'red')
    images = [image]  # Single image in a batch
    predictions = arcface_model.predict(images)
    assert isinstance(predictions, np.ndarray), "Predictions should be a numpy array"
    assert predictions.shape == (1, 512), "Predictions shape should be (1, 512)"
    assert not np.any(np.isnan(predictions)), "Predictions should not contain NaN values"

def test_predict_different_image(arcface_model):
    image1 = Image.new('RGB', (256, 256), color = 'red')
    image2 = Image.new('RGB', (256, 256), color = 'blue')
    images = [image1, image2]  # Batch with different images
    predictions = arcface_model.predict(images)
    assert isinstance(predictions, np.ndarray), "Predictions should be a numpy array"
    assert predictions.shape == (2, 512), "Predictions shape should be (2, 512)"
    assert not np.any(np.isnan(predictions)), "Predictions should not contain NaN values"
