import os
import io
import json
import torch
from PIL import Image
from typing import List, Dict, Any
from ts.torch_handler.base_handler import BaseHandler

TS_IS_RUNNING = bool(os.environ.get("TS_IS_RUNNING"))

if TS_IS_RUNNING:
    from customresnetclassifier import CustomResnetClassifier
else:
    from .customresnetclassifier import CustomResnetClassifier


class ResnetClassifierModelHandler(BaseHandler):
    """
    A dynamic handler for serving different ResNet50-based models
    for facial attribute prediction.
    """

    def initialize(self, context: Any) -> None:
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load index_to_name.json file to get mapping
        # and convert keys to integers
        mapping_file_path = os.path.join(model_dir, "index_to_name.json")
        with open(mapping_file_path) as f:
            self.mapping = json.load(f)
        self.mapping = {int(k): v for k, v in self.mapping.items()}

        # Determine the number of classes from the mapping file
        self.num_classes = len(self.mapping) if len(self.mapping) > 2 else 1

        # Load model weights
        serialized_file = self.manifest["model"]["serializedFile"]
        model_pt_path = os.path.join(model_dir, serialized_file)
        self.model = CustomResnetClassifier(
            weights=model_pt_path, num_classes=self.num_classes, device=self.device
        )

        self.initialized = True

    def _preprocess_one_image(self, req: Dict[str, Any]) -> Image.Image:
        """
        Process one single image.
        """
        image = req.get("data")
        if image is None:
            image = req.get("body")
        # create a stream from the encoded image
        image = Image.open(io.BytesIO(image))
        return image

    def preprocess(self, requests: List[Dict[str, Any]]) -> List[Image.Image]:
        images = [self._preprocess_one_image(req=req) for req in requests]
        return images

    def inference(self, imgs: List[Image.Image]) -> List[Dict[str, float]]:
        # Perform model inference and return the raw outputs
        outputs = self.model.predict_label(imgs, self.mapping)
        return outputs

    def postprocess(
        self, inference_output: List[Dict[str, float]]
    ) -> List[Dict[str, float]]:
        return inference_output
