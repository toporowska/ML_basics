import torch
import numpy as np
import torch.nn as nn
from random import sample
import math
from sklearn.preprocessing import StandardScaler

############# logistic regression class #########################
class LogisticRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LogisticRegression,self).__init__()
        self.linear = nn.Linear(input_size,output_size)
        
    def forward(self,x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted

############# data preparation ############
data = np.loadtxt('data.csv', delimiter=",", dtype=np.float32, skiprows=1)
n_features = data.shape[1]-1
n_samples = data.shape[0]

#train and test sets
train_idx = sample(range(n_samples), math.floor(n_samples*0.9))
test_idx = [i for i in range(n_samples) if i not in train_idx]

#division
train_X = data[train_idx, 0:n_features]
test_X = data[test_idx, 0:n_features]

#scalling
sc = StandardScaler()
train_X = sc.fit_transform(train_X)
test_X = sc.transform(test_X)

#conversion to tensors
train_X = torch.from_numpy(train_X.astype(np.float32))
test_X = torch.from_numpy(test_X.astype(np.float32))
train_Y = torch.from_numpy(data[train_idx, -1].astype(np.float32))
test_Y = torch.from_numpy(data[test_idx, -1].astype(np.float32))


############ model, loss and optimizer ##############
model = nn.Linear( n_features,1)
learning_rate = 0.01
crit = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

model2 = LogisticRegression(n_features,1)
optimizer2 = torch.optim.SGD(model2.parameters(), lr = learning_rate)

############ training loop #################

num_epochs = 1000
for epoch in range(num_epochs):

    # forward and loss for linear
    prediction = model(train_X)
    loss = crit(torch.reshape(prediction,(-1,)), train_Y)

    # forward and loss for logistic
    prediction2 = model2(train_X)
    loss2 = crit(torch.reshape(prediction2,(-1,)), train_Y)
    
    # backward
    loss.backward()
    loss2.backward()
    
    # update
    optimizer.step()
    optimizer2.step()
    
    # zero grad
    optimizer.zero_grad()
    optimizer2.zero_grad()
    
    if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, linear loss = {loss.item():.4f}, logistic loss = {loss2.item():.4f}')
              

############ testing #########

with torch.no_grad():
    prediction = model(test_X)
    prediction = torch.reshape(prediction,(-1,))
    pred_class = prediction.round()
    loss = crit(prediction,test_Y)
    acc = (pred_class == test_Y).sum()/prediction.shape[0]
    print(f'linear: loss:{loss:.4f}, accuracy: {acc:.4f}')

    prediction2 = model2(test_X)
    prediction2 = torch.reshape(prediction2,(-1,))
    pred_class2 = prediction2.round()
    loss2 = crit(prediction2,test_Y)
    acc2 = (pred_class2 == test_Y).sum()/prediction2.shape[0]
    print(f'logistic: loss:{loss2:.4f}, accuracy: {acc2:.4f}')