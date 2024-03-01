from basehandler import DynamicModelHandler

class MockContext:
    def __init__(self, model_dir='models/baldness'):
        self.system_properties = {
            "model_dir": model_dir,
            "gpu_id": "0"  # Assuming a GPU is available; use "cpu" otherwise
        }
        self.manifest = {"model": {"serializedFile": "weights/bald_weights.pth"}}


def load_image_as_request_input(image_path):
    with open(image_path, 'rb') as img_file:
        image_bytes = img_file.read()
    return [{"body": image_bytes}]

def test_handler(image_path):
    # Load the image as a request input
    req_input = load_image_as_request_input(image_path)

    # Initialize handler instance
    handler_instance = DynamicModelHandler()

    # Simulate initializing the handler
    mock_context = MockContext()
    handler_instance.initialize(mock_context)

    # Process the image through the handler workflow
    preprocessed_image = handler_instance.preprocess(req_input)
    result = handler_instance.inference(preprocessed_image)
    final_result = handler_instance.postprocess(result)

    print("Final Result:", final_result)
    
if __name__ == "__main__":
    image_path = 'test_images/baldboi.jpg'  # Specify the path to your test image
    test_handler(image_path)