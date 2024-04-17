import torch
import torch.nn
import torch.nn.functional as F

import numpy as np
import pandas as pd

class NeuralNet(torch.nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()

        self.Linear1 = torch.nn.Linear(3, 4) # input layer
        self.Linear2 = torch.nn.Linear(4, 4) # hidden layer
        self.Linear3 = torch.nn.Linear(4, 1) # output layer

    def forward(self, x):
        x = self.Linear1(x)
        x = self.Linear2(x)
        x = self.Linear3(x)
        return x
    
def main():
    # prepare data
    n_data = 100
    y = np.random.uniform(0, np.pi, n_data) # label
    x1 = y * 1.2
    x2 = y * 0.8
    x3 = y * 0.5
    X = np.array([x1, x2, x3]).T # features

    # training model
    nn = NeuralNet()
    optimizer = torch.optim.Adam(nn.parameters(), lr=0.1)
    MSELoss = torch.nn.MSELoss()

    X = torch.Tensor(X)
    y = torch.Tensor(y)

    epoch = 100
    for i in range(epoch):
        yhat = nn(X)
        loss = MSELoss(yhat.flatten(), y)
        print(f'Epoch {i}   Loss {loss:.3f}')
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # inference
    pred = nn(X)
    print(y)
    print(pred.flatten())
    print(y-pred.flatten())

main()