import torch
import numpy as np
import cv2 as cv
from torchvision.models import resnet34, ResNet34_Weights


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        weight = ResNet34_Weights.DEFAULT
        self.resnet = resnet34(weight)

    def pre_process(self, tensor:torch.Tensor):
        # tensor shape = [1, 256, 256, 3]
        tensor = tensor.permute([0, 3, 1, 2])*(1.0/225.0)

    def forward(self, x):
        # opencv shape为bgr通道，需要进行预处理
        pass