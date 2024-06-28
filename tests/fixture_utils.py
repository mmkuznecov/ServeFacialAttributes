import pytest
from .test_utils import load_image_as_request_input


def create_handler_instance(handler_class):
    """
    Creates a fixture for instantiating and initializing a handler.

    :param handler_class: The handler class to instantiate.
    :return: A fixture that creates and initializes a handler instance.
    """

    @pytest.fixture
    def handler_instance(mock_context):
        # Create an instance of the handler
        handler = handler_class()
        # Initialize the handler with the mock_context
        handler.initialize(mock_context)
        return handler

    return handler_instance


@pytest.fixture
def req_input(image_path):
    """
    Fixture for loading the request input data.

    :param image_path: Path to the image.
    :return: Loaded request input data.
    """
    return load_image_as_request_input(image_path)


@pytest.fixture
def preprocessed_image(handler_instance, req_input):
    """
    Fixture for preprocessing the input data.

    :param handler_instance: Instance of the handler.
    :param req_input: Request input data.
    :return: Preprocessed image.
    """
    return handler_instance.preprocess(req_input)


@pytest.fixture
def inference_output(handler_instance, preprocessed_image):
    """
    Fixture for performing model inference.

    :param handler_instance: Instance of the handler.
    :param preprocessed_image: Preprocessed image.
    :return: Model inference output.
    """
    return handler_instance.inference(preprocessed_image)


# Parametrization


def create_parametrize_mock_context(mock_params):
    """
    Creates a decorator for parametrizing mock_context.

    :param mock_params: Parameters for creating mock_context.
    :return: pytest.mark.parametrize decorator for mock_context.
    """
    return pytest.mark.parametrize("mock_context", mock_params, indirect=True)


def create_parametrize_image_path(paths, param_name="image_path"):
    """
    Creates a decorator for parametrizing the image path.

    :param paths: List of image paths.
    :param param_name: Name of the parameter (default is 'image_path').
    :return: pytest.mark.parametrize decorator for the image path.
    """
    return pytest.mark.parametrize(param_name, paths)
