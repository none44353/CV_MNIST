import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self, transform, data, label):
        super(MNISTDataset, self).__init__() 
        self.transform = transform 
        self.images = data 
        self.labels = label 
    
    def __getitem__(self, idx):
        img = self.images[idx]
        img = self.transform(img)
        label = self.labels[idx]
        return img, label

    def __len__(self):
        return len(self.images)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_data = np.load('train_data.npy')
train_label = np.load('train_label.npy')
test_data = np.load('test_data.npy')
test_label = np.load('test_label.npy')

trainset = MNISTDataset(transform=transform, data=train_data, label=train_label)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)

testset = MNISTDataset(transform=transform, data=test_data, label=test_label)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=0)

classes = ('0', '1', '2', '3', '4','5', '6', '7', '8', '9')

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 2维卷积，输入通道3，输出通道6，卷积核大小5x5
        # 还有其它参数可以设置 (stride, padding)

        # 单通道图像，4个卷积核，卷积核大小5*5
        self.conv1 = nn.Conv2d(3, 4, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(4, 10, 5)

        # fc fully connected，全连接层
        self.fc1 = nn.Linear(8 * 8 * 10, 150)
        self.fc2 = nn.Linear(150, 80)
        self.fc3 = nn.Linear(80, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x)) #28 * 28 * 1 -> 24 * 24 * 4
        x = self.pool(x)          #24 * 24 * 4 -> 12 * 12 * 4
        x = F.relu(self.conv2(x)) #12 * 12 * 4 -> 8 * 8 * 10
    
        x = x.view(-1, 8 * 8 * 10)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 实例化一个网络
net = Net()

import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
running_loss = 0

for epoch in range(50):
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad() 
        #前传
        outputs = net(inputs)
        #计算 loss
        labels = torch.LongTensor(labels)  
        loss = criterion(outputs, labels)
        #反传
        loss.backward()
        #更新
        optimizer.step()#一个统计 training loss 的方法
        running_loss += loss.item()
        if i % 2000 == 1999: #print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
            (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
            PATH = './net50.pth'
            torch.save(net.state_dict(), PATH) #存模型

net = Net()
net.load_state_dict(torch.load('./net.pth')) #加载之前训好的模型参数
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

total, correct = 0, 0
for i in range(10):
    correct += class_correct[i]
    total += class_total[i]

print(correct / total)

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i])), print(100 * class_correct[i] / class_total[i])