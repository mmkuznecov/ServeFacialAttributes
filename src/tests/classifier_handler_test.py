import pytest
from src.handlers.classifiers_handler.classifier_handler import (
    ResnetClassifierModelHandler,
)
from src.utils.test_utils import load_image_as_request_input, mock_context
import os


@pytest.mark.parametrize(
    "mock_context",
    [
        {"model_dir": "models/baldness", "serialized_file": "weights/bald_weights.pth"},
        {"model_dir": "models/beard", "serialized_file": "weights/beard_weights.pth"},
        {
            "model_dir": "models/emotions",
            "serialized_file": "weights/emotion_model.pth",
        },
        {"model_dir": "models/gender", "serialized_file": "weights/gender_model.pth"},
        {
            "model_dir": "models/glasses",
            "serialized_file": "weights/glasses_weights.pth",
        },
        {"model_dir": "models/happiness", "serialized_file": "weights/happy_model.pth"},
        {"model_dir": "models/race", "serialized_file": "weights/race_model.pth"},
    ],
    indirect=True,
)
def test_handler_with_mock(mock_context):
    image_path = os.path.join("test_images", "not_bald.jpg")
    req_input = load_image_as_request_input(image_path)
    handler_instance = ResnetClassifierModelHandler()
    handler_instance.initialize(mock_context)
    preprocessed_image = handler_instance.preprocess(req_input)
    result = handler_instance.inference(preprocessed_image)
    final_result = handler_instance.postprocess(result)
    assert isinstance(final_result, list), "Final result should be a list"
    print(
        "Final Result for model:",
        mock_context.system_properties["model_dir"],
        final_result,
    )
