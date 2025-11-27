from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms

import medmnist
from medmnist import INFO, Evaluator, ChestMNIST, VesselMNIST3D

dataset = ChestMNIST(split="val", download=True, size=28)
dataset_brain = VesselMNIST3D(split="val", download=True, size=28)


data_flag = 'chestmnist'

print(INFO[data_flag]['n_channels'])
dataset.montage(length=1)

DataClass = getattr(medmnist, INFO[data_flag]['python_class'])

# preprocessing
#data_transform = transforms.Compose([
#    transforms.ToTensor(),
#    transforms.Normalize(mean=[.5], std=[.5])
#])

# load the data
train_dataset = DataClass(split='train', download=True)
test_dataset = DataClass(split='test', download=True)

train_dataset.montage(length=20)
