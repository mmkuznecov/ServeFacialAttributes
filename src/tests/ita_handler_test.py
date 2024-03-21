import pytest
from src.handlers.ita_handler.ita_handler import ITAHandler
from src.utils.test_utils import load_image_as_request_input, mock_context


@pytest.mark.parametrize(
    "mock_context",
    [{"model_dir": "models/ita", "serialized_file": "weights/lbfmodel.yaml"}],
    indirect=True,
)
@pytest.mark.parametrize(
    "image_path", ["test_images/bald.jpg", "test_images/not_bald.jpg"]
)
def test_ita_handler(mock_context, image_path):
    request_input = load_image_as_request_input(image_path)
    # Initialize handler instance
    handler = ITAHandler()
    handler.initialize(mock_context)
    preprocessed_input = handler.preprocess(request_input)
    inference_output = handler.inference(preprocessed_input)
    postprocessed_output = handler.postprocess(inference_output)
    # Assertions to validate the output
    assert isinstance(postprocessed_output, list), "Output should be a list"
    assert len(postprocessed_output) > 0, "Output list should not be empty"
    assert "ita_value" in postprocessed_output[0], "Output should contain 'ita_value'"
    assert isinstance(postprocessed_output[0]["ita_value"], list) or isinstance(
        postprocessed_output[0]["ita_value"], float
    ), "'ita_value' should be a list or float"
    # Example of a more specific test, verifying ITA value range if known
    for output in postprocessed_output:
        ita_value = (
            output["ita_value"][0]
            if isinstance(output["ita_value"], list)
            else output["ita_value"]
        )
    print(
        f"Test passed for image: {image_path} with ITA value: {postprocessed_output[0]['ita_value']}"
    )
