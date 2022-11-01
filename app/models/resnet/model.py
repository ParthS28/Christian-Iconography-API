import pandas as pd
import numpy as nps
from tqdm import tqdm

from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torch.nn.functional as F
import torchvision.transforms.functional as FT 

import torchvision.models as models
import torch.nn as nn
import torch.optim as optim

import os
import cv2
import torch
from PIL import Image

class ArtDLClassifier(nn.Module):
  def __init__(self, num_classes):
    super(ArtDLClassifier, self).__init__()
    # Loading the pretrained model
    self.resnet = models.resnet50(pretrained=True)

    self.stage1 = nn.Sequential(self.resnet.conv1, self.resnet.bn1, self.resnet.relu, self.resnet.maxpool,
                                    self.resnet.layer1)
    self.stage2 = nn.Sequential(self.resnet.layer2)
    self.stage3 = nn.Sequential(self.resnet.layer3)
    self.stage4 = nn.Sequential(self.resnet.layer4)

    self.avgpool = self.resnet.avgpool
    self.fc_conv = nn.Conv2d(in_channels = 2048, out_channels = num_classes, kernel_size=1)

    # Setting the trainable params for the optimizer
    self.tr_params = nn.ModuleList([self.stage3, self.stage4, self.avgpool, self.fc_conv])

    self.gradients = None

  #hook for the gradients of the activations
  def activations_hook(self, grad):
    self.gradients = grad

  def forward(self, image):
    # Forward prop
    out = self.stage1(image)
    out = self.stage2(out)
    out = self.stage3(out)
    h = out.register_hook(self.activations_hook)
    out = self.stage4(out)

    # h = out.register_hook(self.activations_hook) ##### Uncomment if you want to get gradients at stage 4

    out = self.avgpool(out)
    
    out = self.fc_conv(out)
    return out

  def trainable_params(self):
    return (list(self.tr_params.parameters()))

  def get_activations_gradient(self):
        return self.gradients
    
  # method for the activation extraction
  def get_activations(self, x):
    out = self.stage1(x)
    out = self.stage2(out)
    return self.stage3(out)
    
    # return self.stage4(out)  ##### Uncomment if you want to get gradients at stage 4
