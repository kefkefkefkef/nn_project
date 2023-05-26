import streamlit as st
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.datasets import CIFAR10
from torchvision import transforms as T
from torchvision import io
import numpy as np
import torchutils as tu
import matplotlib.pyplot as plt
import time
from typing import Tuple

from torchvision.models import resnet18, ResNet18_Weights

model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.load_state_dict(torch.load('../resnet_cats_dogs.py'))
resize = T.Resize((224, 224))
#img = resize(io.read_image('cat.jpg')/255)
device = 'cuda'

st.markdown("# Котики и собачки 🎉")
st.sidebar.markdown("# Котики и собачки 🎉")

input_file = st.file_uploader("Загрузите картинку",type=['jpg'])
img = resize(input_file/255)
#img, true_label = next(iter(train_loader))
img = img.to(device)

pred_class = ('Dog' if model.to(device)(img.unsqueeze(0)).softmax(dim=1).argmax().item()==1 else 'Cat')
#real_class=('Dog' if true_label[0]==1 else 'Cat')

st.write(pred_class)