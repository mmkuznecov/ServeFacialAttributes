import pytest
from unittest.mock import Mock
from ts.context import Context, RequestProcessor
import base64


@pytest.fixture
def mock_context(request):
    """
    Fixture to create a mock context object for testing.

    The fixture accepts parameters to customize the context properties and behavior.
    Parameters can be passed using the `pytest.mark.parametrize` decorator.

    Parameters:
        request: The pytest request object.

    Returns:
        A mock Context object with the specified properties and behavior.
    """
    # Use getattr to safely get request.param
    request_param = getattr(request, "param", {})

    context = Context(
        model_name=request_param.get("model_name", ""),
        model_dir=request_param.get("model_dir", ""),
        manifest=request_param.get("manifest", {}),
        batch_size=request_param.get("batch_size", 1),
        gpu=request_param.get("gpu", "0"),
        mms_version=request_param.get("mms_version", "1.0"),
        limit_max_image_pixels=request_param.get("limit_max_image_pixels", True),
        metrics=request_param.get("metrics", None),
        model_yaml_config=request_param.get("model_yaml_config", None),
    )

    if "serialized_file" in request_param:
        context.manifest["model"] = {"serializedFile": request_param["serialized_file"]}

    # Create a mock request_processor object and set it in the context
    request_processor_mock = Mock(spec=RequestProcessor)
    # Set the side effect for get_request_property to return values from the request_header parameter
    request_processor_mock.get_request_property.side_effect = (
        lambda key: request_param.get("request_header", {}).get(key)
    )

    # Set the request body in the mock request_processor
    request_processor_mock.request_body = request_param.get("request_body", {})

    context.request_processor = [request_processor_mock]

    return context


def load_image_as_request_input(image_path, encode_base64=False):
    with open(image_path, "rb") as img_file:
        data = img_file.read()

    if encode_base64:
        data = base64.b64encode(data).decode("utf-8")

    return [{"body": data}]
