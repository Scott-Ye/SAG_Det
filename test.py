import torch
import os
from run import *
import cv2

net = torch.load('output/epoch4/model.pt').to(device)
with torch.no_grad():
    for index,(images,label) in enumerate(test_loader):
            target = torch.zeros(images.shape[1]).to(torch.long)
            for i in label:
                target[i-1] = 1
            target = target.to(device)
            images = torch.squeeze(images)
            images = images.to(device)
            y = net(images)
            print(y)
            y=int(torch.squeeze(y).argmax())
            print(y in label)
            print(y,label)
