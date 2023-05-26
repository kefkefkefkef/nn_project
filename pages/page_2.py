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
from PIL import Image
#import tensorflow as tf

from torchvision.models import resnet18, ResNet18_Weights
device = 'cpu'

model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.to(device)
model.fc = nn.Linear(512, 1)
model.load_state_dict(torch.load('kotosobaki.pt', map_location=torch.device('cpu')))
model.eval()
resize = T.Resize((224, 224)).cpu()
#img = resize(io.read_image('dog.jpeg')/255)


st.markdown("# Котики и собачки 🎉")
st.sidebar.markdown("# Котики и собачки 🎉")

input_file = st.file_uploader("Загрузите картинку",type=["png", "jpg", "jpeg"])
if (input_file is not None):
    st.write(input_file)
    image = Image.open(input_file)
    #image = Image.open('dog.jpeg')
    img_array = np.array(image)
    new_img = torch.from_numpy(img_array)
    #print(new_img)
    with open('tensor.pt', 'rb') as f:
        input_file = io.BytesIO(f.read())
    new_img = resize(torch.load(input_file, map_location=torch.device('cpu'))/255)
    # = resize(new_img/255)
    #print(new_img)
    #print(new_img.shape)

    pred_class = ('Dog' if model(new_img.unsqueeze(0)).item()>0 else 'Cat')
    st.write(pred_class)
#print(pred_class)