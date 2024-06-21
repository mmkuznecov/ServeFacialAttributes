import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from typing import Union, List, Optional
from PIL import Image
import math
import os

TS_IS_RUNNING = bool(os.environ.get("TS_IS_RUNNING"))

if TS_IS_RUNNING:
    from sixderpnet360_utils import (
        compute_rotation_matrix_from_ortho6d,
        compute_euler_angles_from_rotation_matrices,
    )
else:
    from .sixderpnet360_utils import (
        compute_rotation_matrix_from_ortho6d,
        compute_euler_angles_from_rotation_matrices,
    )


class SixDRepNet360(nn.Module):
    def __init__(self, block: nn.Module, layers: List[int], fc_layers: int = 1) -> None:
        super(SixDRepNet360, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.linear_reg = nn.Linear(512 * block.expansion, 6)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(
        self, block: nn.Module, planes: int, blocks: int, stride: int = 1
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear_reg(x)
        out = compute_rotation_matrix_from_ortho6d(x)
        return out


class HeadPoseEstimator:
    def __init__(self, weights_url: Optional[str] = None) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SixDRepNet360(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3])
        self.load_weights(weights_url)
        self.model.to(self.device)
        self.model.eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def load_weights(self, weights_url: Optional[str]) -> None:
        if weights_url is None:
            saved_state_dict = torch.hub.load_state_dict_from_url(
                "https://cloud.ovgu.de/s/2sP3yLrEwyfSmqC/download/6DRepNet360_300W_LP_AFLW2000.pth"
            )
        else:
            saved_state_dict = torch.load(weights_url)

        if "model_state_dict" in saved_state_dict:
            self.model.load_state_dict(saved_state_dict["model_state_dict"])
        else:
            self.model.load_state_dict(saved_state_dict)

    def process_image(self, image: Union[str, np.ndarray, Image.Image]) -> torch.Tensor:
        if isinstance(image, Image.Image):
            image_tensor = self.transform(image)
        elif isinstance(image, str):
            image = Image.open(image)
            image_tensor = self.transform(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            image_tensor = self.transform(image)
        else:
            raise TypeError("Unsupported image type")
        return image_tensor.to(self.device)

    def predict(
        self, images: Union[str, np.ndarray, List[Union[str, np.ndarray, Image.Image]]]
    ) -> np.ndarray:
        if not isinstance(images, list):
            images = [images]

        batch = torch.stack([self.process_image(img) for img in images])

        with torch.no_grad():
            R_pred = self.model(batch)
            euler_angles = (
                compute_euler_angles_from_rotation_matrices(R_pred) * 180 / np.pi
            )
            return euler_angles.cpu().numpy()
