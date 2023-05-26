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
device = 'cpu'

model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.to(device)
model.fc = nn.Linear(512, 1)
model.cpu().load_state_dict(torch.load('kotosobaki.pt').cpu())
model.eval()
resize = T.Resize((224, 224))
img = resize(io.read_image('dog.jpeg')/255)


st.markdown("# Котики и собачки 🎉")
st.sidebar.markdown("# Котики и собачки 🎉")

input_file = st.file_uploader("Загрузите картинку",type=['jpg'])
if (input_file is not None) and input_file.name.endswith(".jpg"):
    img = resize(input_file/255)
    img = img.to(device)

    pred_class = ('Dog' if model(img.unsqueeze(0)).item()>0 else 'Cat')
    #real_class=('Dog' if true_label[0]==1 else 'Cat')
    st.write(pred_class)
print(pred_class)