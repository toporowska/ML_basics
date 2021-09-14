import torch
import torch.nn as nn
from preprocessing import load

############# neural network class #########################
class Network(nn.Module):
    def __init__(self,input_size,hidden_size, output_size):
        super().__init__()
        self.lin1 = nn.Linear(input_size,hidden_size)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(hidden_size,output_size)
        self.sigmoid = torch.sigmoid
        
    def forward(self,x):
        out = self.lin1(x)
        out = self.relu(out)
        out = self.lin2(out)
        out = self.sigmoid(out)

        return out
        
train_X, train_Y, test_X, test_Y = load()
n_features = train_X.shape[1]

############ model, loss and optimizer ##############
model = Network(n_features, 2, 1)
learning_rate = 0.01
crit = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

############ training loop ######################

num_epochs = 1000
for epoch in range(num_epochs):
    #forward
    pred = model.forward(train_X)
    pred = torch.reshape(pred,(-1,))
    loss = crit(pred, train_Y)
    

    #backward
    loss.backward()

    #update
    optimizer.step()
    optimizer.zero_grad()

    if (epoch+1) % 100 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

################### testing #########################
with torch.no_grad():
    prediction = model(test_X)
    prediction = torch.reshape(prediction,(-1,))
    pred_class = prediction.round()
    loss = crit(prediction,test_Y)
    acc = (pred_class == test_Y).sum()/prediction.shape[0]
    print(f'linear: loss:{loss:.4f}, accuracy: {acc:.4f}')
