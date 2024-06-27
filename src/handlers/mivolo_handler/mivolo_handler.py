import numpy as np
from PIL import Image
import io
import os
import torch
from typing import List, Dict, Any, Union
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from ts.torch_handler.base_handler import BaseHandler

TS_IS_RUNNING = bool(os.environ.get("TS_IS_RUNNING"))

if TS_IS_RUNNING:
    from mi_volo import MiVOLO
else:
    from .mi_volo import MiVOLO


def prepare_batch(
    image_list: List[Image.Image],
    device: torch.device,
    target_size: int = 224,
    mean: List[float] = IMAGENET_DEFAULT_MEAN,
    std: List[float] = IMAGENET_DEFAULT_STD,
) -> torch.Tensor:
    batch_images = []
    for img in image_list:
        img = img.resize((target_size, target_size))
        img = img.convert("RGB")
        img = torch.from_numpy(np.array(img)).float()
        img = img / 255.0
        img = img.permute(2, 0, 1)
        img = (img - torch.tensor(mean)[:, None, None]) / torch.tensor(std)[
            :, None, None
        ]
        batch_images.append(img)

    batch_images = torch.stack(batch_images, dim=0)
    if device:
        batch_images = batch_images.to(device)
    return batch_images


def age_calculation(
    model_output: float, max_age: float, min_age: float, avg_age: float
) -> float:
    result = model_output * (max_age - min_age) + avg_age
    result = round(result, 2)
    return result


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
