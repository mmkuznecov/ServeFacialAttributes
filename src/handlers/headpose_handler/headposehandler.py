from ts.torch_handler.base_handler import BaseHandler
from PIL import Image
import io
from typing import List
import os

TS_IS_RUNNING = bool(os.environ.get("TS_IS_RUNNING"))

if TS_IS_RUNNING:
    from sixdrepnet360 import HeadPoseEstimator
else:
    from .sixdrepnet360 import HeadPoseEstimator


class SixDRepNet360Handler(BaseHandler):
    """
    TorchServe handler for the SixDRepNet360 model.
    """

    def initialize(self, context):
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        serialized_file = self.manifest["model"]["serializedFile"]
        model_pt_path = os.path.join(model_dir, serialized_file)
        self.estimator = HeadPoseEstimator(weights_url=model_pt_path)

    def _preprocess_one_image(self, req):
        """
        Process one single image.
        """
        image = req.get("data")
        if image is None:
            image = req.get("body")
        # create a stream from the encoded image
        image = Image.open(io.BytesIO(image))
        return image

    def preprocess(self, requests):
        images = [self._preprocess_one_image(req=req) for req in requests]
        return images

    def inference(self, img) -> List:
        """Run inference on the preprocessed data."""
        print(type(img))
        print(type(img[0]))
        return self.estimator.predict(img)

    def postprocess(self, inference_output) -> List:
        """Postprocess the inference output."""
        results = []
        for output in inference_output:
            results.append(
                {
                    "yaw": float(output[0]),
                    "pitch": float(output[1]),
                    "roll": float(output[2]),
                }
            )
        return results
