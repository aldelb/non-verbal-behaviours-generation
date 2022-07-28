from logging import exception
import torch
import torch.nn as nn
import torch.nn.functional as F
import constants.constants as constants
from utils.model_parts import Conv, DoubleConv

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def applyLayer(self, x):
        if(constants.layer == "LSTM" or constants.layer == "GRU"):
            x = torch.reshape(x, (x.shape[0], x.shape[2], x.shape[1]))
            x = self.layer(x)[0]
            x = torch.reshape(x, (x.shape[0], x.shape[2], x.shape[1]))
            return x
        else: #CONV
            x = self.layer1(x)
            x = self.layer2(x)
            return x

    def initLayer(self, h_size, i_size, o_size):
        if(constants.layer == "LSTM"):
            self.layer = nn.LSTM(input_size = h_size, hidden_size = h_size, num_layers=2, bidirectional=True, dropout=constants.dropout, batch_first= True)
            self.i_fc = nn.Linear(i_size, h_size)
            self.o_fc = nn.Linear(2 * h_size, o_size)
        elif(constants.layer == "GRU"):
            self.layer = nn.GRU(input_size = h_size, hidden_size = h_size, num_layers=2, bidirectional=True, dropout=constants.dropout, batch_first= True)
            self.i_fc = nn.Linear(i_size, h_size)
            self.o_fc = nn.Linear(2 * h_size, o_size)
        else: #CONV
            self.layer1 = DoubleConv(h_size, int(h_size/2), constants.kernel_size, constants.hidden_size)
            self.layer2 = DoubleConv(int(h_size/2), int(h_size/2), constants.kernel_size)
            self.i_fc = nn.Linear(i_size, h_size)
            self.o_fc = nn.Linear(int(h_size/2), o_size)

class Generator(Model):

    def __init__(self):
        super(Generator, self).__init__()
        p_size = constants.prosody_size
        n_size = constants.noise_size
        i_size = p_size + n_size
        o_size = constants.pose_size + constants.au_size
        h_size = constants.hidden_size
        self.initLayer(h_size, i_size, o_size)

        
    def forward(self, i, n):
        n = n.reshape((n.shape[0], n.shape[2], n.shape[1]))
        n = n.repeat(1, 1, i.size(2))
        x = torch.cat([i, n], dim=1)
        bs = x.shape[0]
        lt = x.shape[2]
        x = x.view(bs*lt, -1)
        x = self.i_fc(x)
        x = x.view(bs, -1, lt)
        x = F.leaky_relu(x, 1e-2)
        x = self.applyLayer(x)
        x = F.leaky_relu(x, 1e-2)
        x = x.view(bs*lt, -1)
        x = self.o_fc(x)
        x = x.view(bs, -1, lt)
        x = torch.sigmoid(x)
        return x

class Discriminator(Model):

    def __init__(self):

        super(Discriminator, self).__init__()
        c_size = constants.prosody_size
        i_size = constants.pose_size + constants.au_size + c_size
        h_size = constants.hidden_size
        o_size = 1

        self.initLayer(h_size, i_size, o_size)

    def forward(self, x, c):
        #c : condition = prosodique features
        #x : real or fake openface data 
        x = torch.cat([x, c], dim=1)
        bs = x.shape[0]
        lt = x.shape[2]
        x = x.view(bs*lt, -1)
        x = self.i_fc(x)
        x = x.view(bs, -1, lt)
        x = F.leaky_relu(x, 1e-2)
        x = self.applyLayer(x)
        x = F.leaky_relu(x, 1e-2)
        x = x.view(bs*lt, -1)
        x = self.o_fc(x)
        x = x.view(bs, -1, lt)
        x = torch.sigmoid(x)
        return x
