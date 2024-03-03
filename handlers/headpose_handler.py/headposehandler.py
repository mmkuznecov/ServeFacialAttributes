from ts.torch_handler.base_handler import BaseHandler
from PIL import Image
import io
from typing import List
from sixdrepnet360 import HeadPoseEstimator
import os


class SixDRepNet360Handler(BaseHandler):
    """
    TorchServe handler for the SixDRepNet360 model.
    """
    def initialize(self, context):
        """Initialize method loads the model and other required elements."""
        super().initialize(context)
        # Instantiate the HeadPoseEstimator with the model's weight URL or path
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        serialized_file = self.manifest['model']['serializedFile']
        model_pt_path = os.path.join(model_dir, serialized_file)
        self.estimator = HeadPoseEstimator(weights_url=model_pt_path)

    def preprocess(self, data) -> List:
        """Preprocess the input data."""
        images = []

        # Process each input data
        for row in data:
            # Retrieve image data from the request
            image = row.get("data") or row.get("body")
            if isinstance(image, str):
                image = Image.open(io.BytesIO(image.encode('utf-8')))
            elif isinstance(image, bytes):
                image = Image.open(io.BytesIO(image))
            else:
                raise ValueError("Unsupported image format")

            image = self.estimator.process_image(image)
            images.append(image)

        return images

    def inference(self, img) -> List:
        """Run inference on the preprocessed data."""
        return self.estimator.predict(img)

    def postprocess(self, inference_output) -> List:
        """Postprocess the inference output."""
        results = []
        for output in inference_output:
            results.append({
                "yaw": float(output[0]),
                "pitch": float(output[1]),
                "roll": float(output[2])
            })
        return results
