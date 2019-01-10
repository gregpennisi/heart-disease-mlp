import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch.optim as optim

#basic multilayer perceptron

class Net(nn.Module):
    def __init__(self): #define the layers and drop
        super(Net, self).__init__()
        self.fc1 = nn.Linear(12,6)
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(6,1)

    def forward(self, x): #define the forward pass
        x = self.fc2(self.drop(F.relu(self.fc1(x))))
        return x

net = Net()
net.zero_grad()

optimizer = optim.Adam(net.parameters())
criterion = nn.MSELoss()

training_values = pd.read_csv('dataset/train_values.csv')
training_values.drop('patient_id', axis=1, inplace=True)
training_values.drop('thal', axis=1, inplace=True)

# ###EXPERIMENTAL DROPS
# training_values.drop('resting_ekg_results', axis=1, inplace=True)

tv_as_numpy = training_values.values

training_labels = pd.read_csv('dataset/train_labels.csv')
training_labels.drop('patient_id', axis=1, inplace=True)
tl_as_numpy = training_labels.values

testing_values = pd.read_csv('dataset/test_values.csv')
testing_values.drop('patient_id', axis=1, inplace=True)
testing_values.drop('thal', axis=1, inplace=True)

# ###EXPERIMENTAL DROPS
# testing_values.drop('resting_ekg_results', axis=1, inplace=True)

testv_as_numpy = testing_values.values

tv_tensor_list = []

for item in enumerate(tv_as_numpy):
    item = item[1]
    A, B, C, D, E, G, H, I, J, K, L, M = iter(item)
    X = Variable(torch.FloatTensor([A, B, C, D, E, G, H, I, J, K, L, M]), requires_grad=True)

# ###EXPERIMENTAL
#     A, B, C, D, E, G, H, I, J, K, L = iter(item)
#     X = Variable(torch.FloatTensor([A, B, C, D, E, G, H, I, J, K, L]), requires_grad=True)

    tv_tensor_list.append(X)
        
tl_tensor_list = []
        
for item in enumerate(tl_as_numpy):
    Y = item[1]
    Y = Variable(torch.FloatTensor([Y[0]]), requires_grad=False)
    tl_tensor_list.append(Y)

testv_tensor_list = []
        
for item in enumerate(testv_as_numpy):
    item = item[1]
    
    At, Bt, Ct, Dt, Et, Gt, Ht, It, Jt, Kt, Lt, Mt = iter(item)
    Xt = Variable(torch.FloatTensor([At, Bt, Ct, Dt, Et, Gt, Ht, It, Jt, Kt, Lt, Mt]), requires_grad=True)
    
# ###EXPERIMENTAL
#     At, Bt, Ct, Dt, Et, Gt, Ht, It, Jt, Kt, Lt = iter(item)
#     Xt = Variable(torch.FloatTensor([At, Bt, Ct, Dt, Et, Gt, Ht, It, Jt, Kt, Lt]), requires_grad=True)

    testv_tensor_list.append(Xt)

for epoch in range(300):
    for i in range(len(tv_tensor_list)):
        X = tv_tensor_list[i]
        Y = tl_tensor_list[i]
        optimizer.zero_grad()
        y_pred = net(X)
        output = criterion(y_pred, Y)
        output.backward()
        optimizer.step()
    if (epoch % 9 == 0.0):
        print("Epoch {} - loss: {}".format(epoch, output))
    elif (epoch == 299):
        print("Final loss: {}".format(output))
