import torch
import torch.nn as nn
from preprocessing import load

############# logistic regression class #########################
class LogisticRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size,output_size)
        
    def forward(self,x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted

############ data loading #################
train_X, train_Y, test_X, test_Y = load()
n_features = train_X.shape[1]


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