import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import configargparse
import cv2
import os
import numpy as np


# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = torch.device(device)


# def config_parser():
#     parser = configargparse.ArgumentParser()
#     parser.add_argument('--data_path', default='./data/images/')
#     parser.add_argument('--hidden_size', type=int, default=256)
#     parser.add_argument('--resolution', type=int, default=512)
#     return parser


# class detector(nn.Module):
#     def __init__(self, hidden_size=512):
#         super(detector, self).__init__()
#         self.resnet = torchvision.models.resnet18()
#         self.resnet.fc = nn.Linear(self.resnet.fc.in_features, hidden_size)
#         self.lstm = torch.nn.LSTM(
#             input_size=hidden_size, hidden_size=hidden_size, bidirectional=True)
#         self.classifier = nn.Sequential(
#             nn.ReLU(),
#             nn.Linear(hidden_size*2, 2)
#         )

#     def forward(self, x):
#         feature = self.resnet(x)
#         hidden = self.lstm(feature)
#         y = self.classifier(hidden[0])
#         return y


# args = config_parser().parse_args()
# preprocess_train = transforms.Compose([transforms.Normalize(mean=138, std=73),
#                                        transforms.RandomResizedCrop(size=(args.resolution, args.resolution)),
#                                        transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), 
#                                        transforms.RandomRotation(degrees=90)])
# preprocess_test = transforms.Compose([transforms.Normalize(
#     mean=138, std=73), transforms.Resize(size=(args.resolution, args.resolution))])


# class dataset(Dataset):
#     def __init__(self, txt_name, mode):
#         super(dataset, self).__init__()
#         self.path = []
#         self.labels = []
#         self.mode = mode
#         with open(args.data_path+txt_name, 'r') as f:
#             for line in f.readlines():
#                 self.path.append(args.data_path+line.split('\t')[0])
#                 self.labels.append(line.split('\t')[1])

#     def __getitem__(self, index):
#         images = []
#         for img_path in os.listdir(self.path[index]):
#             image = cv2.imread(self.path[index]+'/'+img_path).astype('float32')
#             image = np.transpose(image,(2,1,0))
#             image = torch.tensor(image)
#             if self.mode == 0:
#                 images.append(preprocess_train(image))
#             if self.mode == 1:
#                 images.append(preprocess_test(image))
#             label = []
#             for x in self.labels[index].split(','):
#                 label.append(int(x))
#         images = np.array(images)
#         return torch.tensor(images), label

#     def __len__(self):
#         return len(self.labels)


# train_loader = DataLoader(dataset('train.txt',mode=0),batch_size=1)
# test_loader = DataLoader(dataset('test.txt',mode=1),batch_size=1)


# from sklearn.model_selection import train_test_split

# with open('./data/label.txt','r') as f:
#     data = f.readlines()
#     data_train,data_test = train_test_split(data,test_size=0.2)
# with open('./data/images/train.txt','w') as f:
#     f.writelines(data_train)
# with open('./data/images/test.txt','w') as f:
#     f.writelines(data_test)

# import pandas as pd
# path = 'D:\\work2\\LLM_medicine\\ACC数据索引\\'
# df = pd.read_csv(path+'pacc_index.csv')
# with open('data/label.txt','a') as f:
#     for id,index in zip(df['路径'],df['SShTSE矢位索引']):
#         index = str(index)
#         f.write(id+'\t'+index+'\n')
x=[]
for dir in os.listdir('./data/images/'):
    for img in os.listdir('./data/images/'+dir):
        image = cv2.imread('./data/images/'+dir+'/'+img)
        x.append(cv2.resize(image,(512,512)))
x=np.array(x)
print(x.mean(),x.std())