import pytest
import torch
import json
from PIL import Image
import os
from src.handlers.classifiers_handler.customresnetclassifier import CustomResnetClassifier

@pytest.fixture
def bald_weights_path():
    return "models/baldness/weights/bald_weights.pth"

@pytest.fixture
def race_weights_path():
    return "models/race/weights/race_model.pth"

@pytest.fixture
def bald_mapping():
    with open("models/baldness/index_to_name.json", "r") as f:
        mapping = json.load(f)
    mapping = {int(k): v for k, v in mapping.items()}
    return mapping

@pytest.fixture
def race_mapping():
    with open("models/race/index_to_name.json", "r") as f:
        mapping = json.load(f)
    mapping = {int(k): v for k, v in mapping.items()}
    return mapping

@pytest.fixture
def bald_classifier(bald_weights_path):
    return CustomResnetClassifier(weights=bald_weights_path, num_classes=1)

@pytest.fixture
def race_classifier(race_weights_path):
    return CustomResnetClassifier(weights=race_weights_path, num_classes=7)  # num_classes для race модели

@pytest.fixture
def sample_image_path():
    return os.path.join("tests/test_images", "not_bald.jpg")

@pytest.fixture
def sample_image(sample_image_path):
    return Image.open(sample_image_path)

def test_load_model(bald_weights_path):
    classifier = CustomResnetClassifier(weights=bald_weights_path, num_classes=1)
    assert isinstance(classifier.model, torch.nn.Module), "Model should be an instance of torch.nn.Module"

def test_process_image(sample_image):
    classifier = CustomResnetClassifier(weights="models/baldness/weights/bald_weights.pth", num_classes=1)
    transformed_image = classifier.process_image(sample_image)
    assert isinstance(transformed_image, torch.Tensor), "Transformed image should be a tensor"
    assert transformed_image.shape == (3, 224, 224), "Transformed image shape should be (3, 224, 224)"

def test_predict_single_image_bald(bald_classifier, sample_image):
    prediction = bald_classifier.predict(sample_image)
    assert isinstance(prediction, torch.Tensor), "Prediction should be a tensor"
    assert prediction.shape == (1, bald_classifier.num_classes), f"Prediction shape should be (1, {bald_classifier.num_classes})"

def test_predict_batch_images_bald(bald_classifier, sample_image):
    images = [sample_image for _ in range(4)]  # Create a batch of 4 images
    predictions = bald_classifier.predict(images)
    assert isinstance(predictions, torch.Tensor), "Predictions should be a tensor"
    assert predictions.shape == (4, bald_classifier.num_classes), f"Predictions shape should be (4, {bald_classifier.num_classes})"

def test_predict_label_bald(bald_classifier, sample_image, bald_mapping):
    predictions = bald_classifier.predict_label(sample_image, bald_mapping)
    assert isinstance(predictions, list), "Predictions should be a list"
    assert len(predictions) == 1, "There should be one prediction for a single image"
    assert isinstance(predictions[0], dict), "Each prediction should be a dictionary"
    assert "not_bald" in predictions[0] and "bald" in predictions[0], "Predictions should contain 'not_bald' and 'bald' labels"

def test_predict_label_batch_bald(bald_classifier, sample_image, bald_mapping):
    images = [sample_image for _ in range(4)]  # Create a batch of 4 images
    predictions = bald_classifier.predict_label(images, bald_mapping)
    assert isinstance(predictions, list), "Predictions should be a list"
    assert len(predictions) == 4, "There should be one prediction for each image in the batch"
    for prediction in predictions:
        assert isinstance(prediction, dict), "Each prediction should be a dictionary"
        assert "not_bald" in prediction and "bald" in prediction, "Predictions should contain 'not_bald' and 'bald' labels"

def test_predict_single_image_race(race_classifier, sample_image):
    prediction = race_classifier.predict(sample_image)
    assert isinstance(prediction, torch.Tensor), "Prediction should be a tensor"
    assert prediction.shape == (1, race_classifier.num_classes), f"Prediction shape should be (1, {race_classifier.num_classes})"

def test_predict_batch_images_race(race_classifier, sample_image):
    images = [sample_image for _ in range(4)]  # Create a batch of 4 images
    predictions = race_classifier.predict(images)
    assert isinstance(predictions, torch.Tensor), "Predictions should be a tensor"
    assert predictions.shape == (4, race_classifier.num_classes), f"Predictions shape should be (4, {race_classifier.num_classes})"

def test_predict_label_race(race_classifier, sample_image, race_mapping):
    predictions = race_classifier.predict_label(sample_image, race_mapping)
    assert isinstance(predictions, list), "Predictions should be a list"
    assert len(predictions) == 1, "There should be one prediction for a single image"
    assert isinstance(predictions[0], dict), "Each prediction should be a dictionary"
    for label in race_mapping.values():
        assert label in predictions[0], f"Predictions should contain '{label}' label"

def test_predict_label_batch_race(race_classifier, sample_image, race_mapping):
    images = [sample_image for _ in range(4)]  # Create a batch of 4 images
    predictions = race_classifier.predict_label(images, race_mapping)
    assert isinstance(predictions, list), "Predictions should be a list"
    assert len(predictions) == 4, "There should be one prediction for each image in the batch"
    for prediction in predictions:
        assert isinstance(prediction, dict), "Each prediction should be a dictionary"
        for label in race_mapping.values():
            assert label in prediction, f"Predictions should contain '{label}' label"
