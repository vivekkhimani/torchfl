'''
Federated Learning for FEMNIST
'''
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import os
import numpy as np
import math
import functions

import csv

class myModel(nn.Module):
    def __init__(self, image_size, num_classes):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size= (5,5), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size= (5,5), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2))

        new_size = math.floor((image_size - 2)/2 + 1) #layer 1
        new_size = math.floor((new_size - 2)/2 + 1) #layer 2
   
        self.layer3 = nn.Sequential( 
            nn.Linear(in_features=new_size*new_size*64, out_features=2048),
            nn.ReLU(),
            nn.Linear(in_features=2048, out_features=num_classes))
            #nn.LogSoftmax())
        
    def forward(self, input_data):
        input_data = functions.make2d(input_data)
        input_data = input_data.float()
        out1 = self.layer1(input_data.unsqueeze(1)) 
        out2 = self.layer2(out1)
        out2 = out2.view(out2.shape[0], -1)
        out3 = self.layer3(out2)
            
        return out3

class FEMNIST(Dataset):
    def __init__(self, root_path, client_order, train=True):
        if train:
            filename = f'{root_path}train/user{client_order+1}.csv'
        else:
            filename = f'{root_path}test/user{client_order+1}.csv'
        
        rows = []
        self.x = []
        self.y = []
        with open(filename, 'r') as file:
            csvreader = csv.reader(file)
            header = next(csvreader)
            for row in csvreader:
                rows.append(row)
                x = list(map(float, row[0:-1]))
                self.x.append(x)
                self.y.append(int(row[-1]))
            #self.z = by_class(self.y)
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return torch.tensor(self.x[idx], dtype=torch.float), torch.tensor(self.y[idx])

class ClientModel:
    def __init__(self, client, root_path=None):
        self.myID = client
        self.root_path = root_path
    
    def load_data(self):
        train_dataset = FEMNIST(self.root_path, self.myID, train=True)
        self.samples = len(train_dataset)
        self.train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        test_dataset = FEMNIST(self.root_path, self.myID, train=False)
        self.test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    def create_model(self):
        self.client_model = myModel(28, 62)
        self.criterion = nn.CrossEntropyLoss()
        self.lr = 1e-3
        self.optimizer = Adam(self.client_model.parameters(), lr=self.lr)
        self.epoch = 0

    def new_task(self, message):
        if message['0'] == 'new':
            self.epoch += 1
            client_weights = message['1']
            self.client_model.load_state_dict(client_weights)
            self.client_model.eval()

            data_iter = iter(self.train_loader)
            len_dataloader = len(self.train_loader)

            epoch_loss = 0
            epoch_acc = 0
            i = 0
            while i < len_dataloader:
                self.optimizer.zero_grad()
                self.client_model.zero_grad()
                
                data = data_iter.next()
                img, label= data
                class_output= self.client_model(input_data=img)
                
                err = self.criterion(class_output, label)
                acc = functions.categorical_accuracy(class_output, label)

                err.backward()
                self.optimizer.step()

                epoch_loss += err.item()
                epoch_acc += acc.item() 
                i += 1
            
            loss_all = epoch_loss / len_dataloader
            acc_all =  epoch_acc / len_dataloader
            print(f'\tClient{self.myID} - [EPOCH {self.epoch}]:   Train Loss: {loss_all:.3f} | class Acc: {acc_all*100:.2f}%')
            msg={'weights': self.client_model.state_dict(),
                 'samples': self.samples}
                
        else:
            client_weights = message['1']
            self.client_model.load_state_dict(client_weights)
            self.client_model.eval()
            
            data_iter = iter(self.test_loader)
            len_dataloader = len(self.test_loader)

            test_loss, test_acc = functions.evaluate(self.client_model, self.test_loader, self.criterion) 
            print(f'Client{self.myID} - Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
            msg = {}
        return msg # synchronous

    

