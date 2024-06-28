from PIL import Image
from src.handlers.classifiers_handler.classifier_handler import (
    ResnetClassifierModelHandler,
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
    {"model_dir": "models/baldness", "serialized_file": "weights/bald_weights.pth"},
    {"model_dir": "models/beard", "serialized_file": "weights/beard_weights.pth"},
    {"model_dir": "models/emotions", "serialized_file": "weights/emotion_model.pth"},
    {"model_dir": "models/gender", "serialized_file": "weights/gender_model.pth"},
    {"model_dir": "models/glasses", "serialized_file": "weights/glasses_weights.pth"},
    {"model_dir": "models/happiness", "serialized_file": "weights/happy_model.pth"},
    {"model_dir": "models/race", "serialized_file": "weights/race_model.pth"},
]

IMAGE_PATHS = ["tests/test_images/not_bald.jpg"]

parametrize_mock_context = create_parametrize_mock_context(MOCK_PARAMS)
parametrize_image_path = create_parametrize_image_path(IMAGE_PATHS)

handler_instance = create_handler_instance(ResnetClassifierModelHandler)


@parametrize_mock_context
@parametrize_image_path
def test_preprocess(handler_instance, req_input):
    preprocessed_image = handler_instance.preprocess(req_input)
    assert len(preprocessed_image) == 1
    assert isinstance(preprocessed_image[0], Image.Image)
    print("Preprocessed data:", preprocessed_image)


@parametrize_mock_context
@parametrize_image_path
def test_inference(handler_instance, preprocessed_image):
    result = handler_instance.inference(preprocessed_image)
    assert isinstance(result, list), "Inference result should be a list"
    print("Inference result:", result)


@parametrize_mock_context
@parametrize_image_path
def test_postprocess(handler_instance, inference_output):
    final_result = handler_instance.postprocess(inference_output)
    assert isinstance(final_result, list), "Final result should be a list"
    print("Postprocessed data:", final_result)
