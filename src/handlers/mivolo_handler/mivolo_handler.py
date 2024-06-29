import numpy as np
from PIL import Image
import io
import os
import torch
from typing import List, Dict, Any
from ts.torch_handler.base_handler import BaseHandler

TS_IS_RUNNING = bool(os.environ.get("TS_IS_RUNNING"))

if TS_IS_RUNNING:
    from mi_volo import MiVOLO, prepare_batch, age_calculation
else:
    from .mi_volo import MiVOLO, prepare_batch, age_calculation


class MiVOLOHandler(BaseHandler):

    def initialize(self, context: Any) -> None:
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        serialized_file = self.manifest["model"]["serializedFile"]
        model_pt_path = os.path.join(model_dir, serialized_file)

        self.model = MiVOLO(model_pt_path, self.device)

        self.max_age = self.model.meta.max_age
        self.min_age = self.model.meta.min_age
        self.avg_age = self.model.meta.avg_age

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

    def preprocess(self, requests: List[Dict[str, Any]]) -> torch.Tensor:
        images = [self._preprocess_one_image(req=req) for req in requests]
        image_tensor = prepare_batch(images, self.device)
        return image_tensor

    def inference(self, image_tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            model_outputs = self.model.inference(image_tensor)
        return model_outputs

    def postprocess(self, model_outputs: torch.Tensor) -> List[Dict[str, float]]:
        ages = [
            age_calculation(float(age), self.max_age, self.min_age, self.avg_age)
            for age in model_outputs
        ]
        results = [{"age": age} for age in ages]
        return results
