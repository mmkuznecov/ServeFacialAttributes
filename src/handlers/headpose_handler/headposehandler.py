from ts.torch_handler.base_handler import BaseHandler
from PIL import Image
import io
import numpy as np
from typing import List, Any
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

    def initialize(self, context: Any) -> None:
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        serialized_file = self.manifest["model"]["serializedFile"]
        model_pt_path = os.path.join(model_dir, serialized_file)
        self.estimator = HeadPoseEstimator(weights_url=model_pt_path)

    def _preprocess_one_image(self, req: Any) -> Image.Image:
        """
        Process one single image.
        """
        image = req.get("data")
        if image is None:
            image = req.get("body")
        # create a stream from the encoded image
        image = Image.open(io.BytesIO(image))
        return image

    def preprocess(self, requests: List[Any]) -> List[Image.Image]:
        images = [self._preprocess_one_image(req=req) for req in requests]
        return images

    def inference(self, img: List[Image.Image]) -> List:
        """Run inference on the preprocessed data."""
        return self.estimator.predict(img)

    def postprocess(self, inference_output: np.ndarray) -> List[dict]:
        """Postprocess the inference output."""
        results = []
        for output in inference_output:
            results.append(
                {
                    "pitch": float(output[0]),
                    "yaw": float(output[1]),
                    "roll": float(output[2]),
                }
            )
        return results
