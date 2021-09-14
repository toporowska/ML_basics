import torch
import torch.nn as nn
from preprocessing import load

train_X, train_Y, test_X, test_Y = load()
n_features = train_X.shape[1]

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