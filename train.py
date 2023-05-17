import argparse
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch
from torch import nn, optim
from torchvision import datasets, transforms
import torchvision.models as models

import json

def get_args():
    parser = argparse.ArgumentParser(description="training flower classification")
    parser.add_argument('dir', default="flower_data/train", type=str, help="data directory")
    parser.add_argument('--save-dir', default='data', type=str)
    parser.add_argument('--arch', default='densenet', type=str, choices=['densenet', 'vgg', 'resnet', 'alexnet'])
    parser.add_argument('--learning-rate', default=0.01, type=float)
    parser.add_argument('--hidden-units', default=512, type=int)
    parser.add_argument('--epochs', default=25, type=int)

    return parser.parse_args()

def training_data():
  data_dir = get_args().dir

  data_transforms = transforms.Compose([
      transforms.Resize(256),
      transforms.RandomRotation(60),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
  ])

  image_dataset =  datasets.ImageFolder(data_dir, transform=data_transforms)
  data_loader = torch.utils.data.DataLoader(image_dataset, batch_size=1024, shuffle=True)
  return data_loader

def create_model():
  models_map = {
     "densenet": models.densenet121,
     "vgg": models.vgg11,
     "resnet":models.resnet34,
     "alexnet": models.alexnet,
 }

  model = models_map[get_args().arch](pretrained=True)
  for param in model.parameters():
    param.requires_grad = False

  return model

def main():
    print(create_model())

if __name__ == "__main__":
    main()
