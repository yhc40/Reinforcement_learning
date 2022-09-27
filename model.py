from turtle import forward
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class DQN_withConv(nn.Module):

    def __init__(self):
        super().__init__()
        self.kernel = None
        self.filters = None 
        self.conv1 = nn.Conv2d(18,18)
        self.flatten = nn.Flatten()
        self.pool1 = nn.MaxPool2d(19,20)

    def forward(self,x):
        return x

    def save_model(self,filename):
        torch.save(self.state_dict(), filename)


class DQN(nn.Module):
    
    def __init__(self,input_shape):
        super().__init__()
        self.linear_complex1 = nn.Linear(248,512)
        self.linear_complex2 = nn.Linear(512,248)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(248)
        self.linear1 = nn.Linear(input_shape,64)
        self.linear2 = nn.Linear(64,248)
        self.linear3 = nn.Linear(248,2)
    
    def forward(self,x):
        x= self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.leaky_relu(x)
        x = self.bn1(F.relu(self.linear_complex1(x)))
        x = self.bn2(F.relu(self.linear_complex2(x)))
        x = self.linear3(x)
        return x 

    def save_model(self,file_name):
        torch.save(self.state_dict(), file_name)

class DQN_Trainer:
    def __init__(self,model,target,lr,gamma):
        self.model = model
        self.target_net = target
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.loss = []
    
    def train_step(self,state,action,reward,new_state,done):
        state = torch.tensor(state, dtype=torch.float)
        new_state = torch.tensor(new_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        self.model.train()
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            new_state = torch.unsqueeze(new_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        pred = self.model(state)
        target = pred.clone()

        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.target_net(new_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
        self.loss.append(loss.item()/len(done))
