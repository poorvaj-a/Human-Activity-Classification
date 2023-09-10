import torch

from data import getData
from torch import nn
import numpy as np
import re
import time
import math
from torch.utils.data import Dataset,DataLoader
import numpy as np 
from sklearn.model_selection import train_test_split


Data,Data_hand,Data_chest,Data_ankle = getData()
Y = Data[:,1]
X = Data[:,2:]
input_size = len(X[0]) # 40 features
output_size = 25 # 25 classes
hidden_size = 100

start = time.time()

class Model(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(Model,self).__init__()
        self.hidden1 = nn.Linear(input_size,hidden_size)
        self.relu = nn.LeakyReLU()
        self.hidden2 = nn.Linear(hidden_size,hidden_size)
        self.output = nn.Linear(hidden_size,output_size)
    def forward(self,x):
        x = self.hidden1(x)
        x = self.relu(x)
        x = self.hidden2(x)
        x = self.relu(x)
        x = self.output(x)
        return x

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=1)

class data(Dataset):
    def __init__(self):
        self.x = torch.from_numpy(X_train)
        self.y = torch.from_numpy(y_train)
        self.len = len(self.x)
    def __getitem__(self,index):
        return self.x[index],self.y[index]
    def __len__(self):
        return self.len

dataset = data()

dataloader = DataLoader(dataset=dataset,batch_size=100,shuffle=True)
dataiter = iter(dataloader)
data = dataiter.next()

features, labels = data

model = Model(input_size,hidden_size,output_size)



class test(Dataset):
    def __init__(self):
        self.x = torch.from_numpy(X_test)
        self.y = torch.from_numpy(y_test)
        self.len = len(self.x)
    def __getitem__(self,index):
        return self.x[index],self.y[index]
    def __len__(self):
        return self.len

test_dataset = test()

test_dataloader = DataLoader(dataset=test_dataset,batch_size=100,shuffle=True)
test_dataiter = iter(test_dataloader)
test_data = test_dataiter.next()

num_epoch = 20

total_samples = len(dataset)
n_iterations = math.ceil(total_samples/100)

l = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)
ans =0
for i, (features,labels) in enumerate(test_dataloader):
        output = model(features.float())
        for k in range(len(output)):
            if torch.argmax(output[k]) == labels[k]:
                ans = ans + 1
print(f'Test set accuracy = {ans/len(X_test)}')

for epoch in range(num_epoch):
    lossing = 0
    for i, (features,labels) in enumerate(dataloader):
        output = model(features.float())
        loss = l(output,labels.long())
        lossing = lossing + loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    ans = 0
    for i, (features,labels) in enumerate(test_dataloader):
        output = model(features.float())
        for k in range(len(output)):
            if torch.argmax(output[k]) == labels[k]:
                ans = ans + 1
    print(f'Test set accuracy = {ans/len(X_test)}')
    print(f'{time.time()-start}')
    print(f'epoch {epoch+1}/{num_epoch}, loss = {lossing/len(dataloader):.4f}')

ans = 0
for i, (features,labels) in enumerate(dataloader):
    output = model(features.float())
    for k in range(len(output)):
        if torch.argmax(output[k]) == labels[k]:
            ans = ans + 1
print(f'Training set accuracy = {ans/len(X_train)}')
torch.save(model.state_dict(), 'neural.pth')
class test(Dataset):
    def __init__(self):
        self.x = torch.from_numpy(X_test)
        self.y = torch.from_numpy(y_test)
        self.len = len(self.x)
    def __getitem__(self,index):
        return self.x[index],self.y[index]
    def __len__(self):
        return self.len

test_dataset = test()

test_dataloader = DataLoader(dataset=test_dataset,batch_size=100,shuffle=True)
test_dataiter = iter(test_dataloader)
test_data = test_dataiter.next()
ans = 0
for i, (features,labels) in enumerate(test_dataloader):
    output = model(features.float())
    for k in range(len(output)):
        if torch.argmax(output[k]) == labels[k]:
            ans = ans + 1
print(f'Test set accuracy = {ans/len(X_test)}')

print(time.time() - start)