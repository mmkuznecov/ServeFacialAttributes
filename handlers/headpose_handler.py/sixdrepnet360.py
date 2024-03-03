import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import cv2
from typing import Union, List
from PIL import Image
import sixderpnet360_utils
import math

class SixDRepNet360(nn.Module):
    def __init__(self, block, layers, fc_layers=1):
        self.inplanes = 64
        super(SixDRepNet360, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)

        self.linear_reg = nn.Linear(512*block.expansion,6)
      
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
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
        out = sixderpnet360_utils.compute_rotation_matrix_from_ortho6d(x)

        return out
    


    
class HeadPoseEstimator:
    def __init__(self, weights_url = None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SixDRepNet360(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3])
        self.load_weights(weights_url)
        self.model.to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_weights(self, weights_url: str):
        if weights_url is None:
            saved_state_dict = torch.hub.load_state_dict_from_url("https://cloud.ovgu.de/s/2sP3yLrEwyfSmqC/download/6DRepNet360_300W_LP_AFLW2000.pth")
        else:
            saved_state_dict = torch.load(weights_url)

        if 'model_state_dict' in saved_state_dict:
            self.model.load_state_dict(saved_state_dict['model_state_dict'])
        else:
            self.model.load_state_dict(saved_state_dict)

    def process_image(self, image: Union[str, np.ndarray]) -> torch.Tensor:
        if isinstance(image, str):
            image = cv2.imread(image)
        if isinstance(image, np.ndarray):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        return self.transform(image)

    def predict(self, images: Union[str, np.ndarray, List[Union[str, np.ndarray]]]) -> Union[np.ndarray, List[np.ndarray]]:
        if not isinstance(images, list):
            images = [images]
        
        batch = torch.stack([self.process_image(img) for img in images]).to(self.device)
        
        with torch.no_grad():
            R_pred = self.model(batch)
            euler_angles = sixderpnet360_utils.compute_euler_angles_from_rotation_matrices(R_pred) * 180 / np.pi
            return euler_angles.cpu().numpy()
        
        
def test():
    
    head_pose_estimator = HeadPoseEstimator()
    
    image_paths = ['face_img.jpg', 'face_img.jpg']
    
    poses = head_pose_estimator.predict(image_paths)
    
    for pose in poses:
        yaw, pitch, roll = pose
        print(f"Yaw: {yaw}, Pitch: {pitch}, Roll: {roll}")

if __name__ == "__main__":
    
    test()
    