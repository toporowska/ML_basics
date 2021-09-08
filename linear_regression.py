import torch
import numpy as np
import torch.nn as nn
from random import sample
import math
from sklearn.preprocessing import StandardScaler

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

############ training loop #################

num_epochs = 100
for epoch in range(num_epochs):

    # forward and loss
    prediction = model(train_X)
    loss = crit(torch.reshape(prediction,(-1,)), train_Y)
    
    # backward
    loss.backward()
    
    # update
    optimizer.step()
    
    # zero grad
    optimizer.zero_grad()
    
    if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')
              

############ testing #########

with torch.no_grad():
    prediction = model(test_X)
    prediction = torch.reshape(prediction,(-1,))
    pred_class = prediction.round()
    acc = (pred_class == test_Y).sum()/prediction.shape[0]
    loss = crit(prediction,test_Y)
    print(f'loss:{loss:.4f}, accuracy:{acc:.4f}')