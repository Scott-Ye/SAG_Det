import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader,Subset
import configargparse
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

# 请使用CUDA_VISIBLE_DEVICES指定显卡

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)


def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--data_path', default='./data/images/')
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--resolution', type=int, default=600)
    parser.add_argument('--epochs',type=int,default=5)
    parser.add_argument('--i_save',type=int,default=1)
    parser.add_argument('--lr',type=float,default=1e-4)
    return parser


class detector(nn.Module):
    def __init__(self, hidden_size=512):
        super(detector, self).__init__()
        self.resnet = torchvision.models.resnet18()
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, hidden_size)
        self.lstm = torch.nn.LSTM(
            input_size=hidden_size, hidden_size=hidden_size, num_layers=2, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size*2, 1)
        )

    def forward(self, x):
        feature = self.resnet(x)
        hidden = self.lstm(feature)
        y = self.classifier(hidden[0])
        return y

args = config_parser().parse_args()

def class_weight():
    data = []
    with open(args.data_path+'train.txt','r') as f:
        for line in f.readlines():
            dir = line.split('\t')[0]
            for i in range(len(os.listdir(args.data_path+dir))):
                if str(i) in line.split('\t')[1].strip().split(','):
                    data.append(1)
                else:
                    data.append(0)
    classes = [0,1]
    weights = compute_class_weight(class_weight='balanced',classes=classes,y=data)
    return torch.tensor(weights,dtype=torch.float).to(device)
    
preprocess_train = transforms.Compose([transforms.Normalize(mean=22.95364, std=45.944538537885734),
                                       transforms.RandomResizedCrop(size=(args.resolution, args.resolution)),
                                       transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), 
                                       transforms.RandomRotation(degrees=90)])
preprocess_test = transforms.Compose([transforms.Normalize(
    mean=22.95364, std=45.944538537885734), transforms.Resize(size=(args.resolution, args.resolution))])


class dataset(Dataset):
    def __init__(self, txt_name, mode):
        super(dataset, self).__init__()
        self.path = []
        self.labels = []
        self.mode = mode
        with open(args.data_path+txt_name, 'r') as f:
            for line in f.readlines():
                self.path.append(args.data_path+line.split('\t')[0])
                self.labels.append(line.split('\t')[1])

    def __getitem__(self, index):
        images = []
        for img_path in os.listdir(self.path[index]):
            image = cv2.imread(self.path[index]+'/'+img_path).astype('float32')
            image = np.transpose(image,(2,1,0))
            image = torch.tensor(image)
            if self.mode == 0:
                images.append(preprocess_train(image))
            if self.mode == 1:
                images.append(preprocess_test(image))
            label = []
            for x in self.labels[index].split(','):
                label.append(int(x))
        images = np.array(images)
        return torch.tensor(images), label

    def __len__(self):
        return len(self.labels)


train_loader = DataLoader(dataset('train.txt',mode=0),batch_size=1)
test_loader = DataLoader(dataset('test.txt',mode=1),batch_size=1)
train_sub_loader = DataLoader(Subset(dataset('train.txt',mode=1),np.linspace(0,46,12).astype('uint8')),batch_size=1)

def test(dataloader,net):
    n=0
    m=0
    with torch.no_grad():
        for index,(images,label) in enumerate(dataloader):
            target = torch.zeros(images.shape[1]).to(torch.float32)
            for i in label:
                target[i-1] = 1
            target = target.to(device)
            images = torch.squeeze(images)
            images = images.to(device)
            y = net(images)
            y=int(torch.squeeze(y).argmax())
            n+=1
            m+=int(y in label)
        return m/n

def main():
    net = detector(hidden_size=args.hidden_size).to(device)
    optimizer = torch.optim.Adam(net.parameters(),lr=args.lr)
    lr_sheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs)
    loss_fun = nn.MSELoss()
    train_loss = []
    train_acc = []
    test_acc = []
    for epoch in range(args.epochs):
        for index,(images,label) in enumerate(tqdm(train_loader)):
            net.train()
            target = torch.zeros(images.shape[1]).to(torch.float32)
            for i in label:
                target[i-1] = 1
            target = target.to(device)
            images = torch.squeeze(images)
            images = images.to(device)
            y = torch.squeeze(net(images))
            loss = loss_fun(y,target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss.append(float(loss))
        lr_sheduler.step()
        if (epoch+1)%args.i_save == 0:
            net.eval()
            path = './output/epoch'+str(epoch)
            try:
                os.mkdir(path)
            except:
                pass
            train_acc.append(test(train_sub_loader,net))
            test_acc.append(test(test_loader,net))
            pd.DataFrame({'train':train_acc,'test':test_acc}).to_csv(path+'/acc.csv')
            plt.plot(train_loss)
            plt.savefig(path+'/loss.png')
            torch.save(net,path+'/model.pt')

if __name__=='__main__':
    main()


