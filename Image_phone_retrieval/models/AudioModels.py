# Modified from https://github.com/dharwath/DAVEnet-pytorch.git
import torch
import torch.nn as nn
import torch.nn.functional as F

        
# class for making multi headed attenders.
class multi_attention(nn.Module):
    def __init__(self, in_size, hidden_size, n_heads):
        super(multi_attention, self).__init__()
        self.att_heads = nn.ModuleList()
        for x in range(n_heads):
            self.att_heads.append(attention(in_size, hidden_size))
    def forward(self, input):
        out, self.alpha = [], []
        for head in self.att_heads:
            o = head(input)
            out.append(o) 
            # save the attention matrices to be able to use them in a loss function
            self.alpha.append(head.alpha)
        # return the resulting embedding 
        return torch.cat(out, 1)

class attention(nn.Module):
    def __init__(self, in_size, hidden_size):
        super(attention, self).__init__()
        self.hidden = nn.Linear(in_size, hidden_size)
        nn.init.orthogonal(self.hidden.weight.data)
        self.out = nn.Linear(hidden_size, in_size)
        nn.init.orthogonal(self.hidden.weight.data)
        self.softmax = nn.Softmax(dim = 1)
    def forward(self, input):
        # calculate the attention weights
        self.alpha = self.softmax(self.out(nn.functional.tanh(self.hidden(input))))
        # apply the weights to the input and sum over all timesteps
        x = torch.sum(self.alpha * input, 1)
        # return the resulting embedding
        return x 


class Davenet(nn.Module):
    def __init__(self, embedding_dim=1024):
        super(Davenet, self).__init__()
        self.embedding_dim = embedding_dim
        self.batchnorm1 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, 128, kernel_size=(40,1), stride=(1,1), padding=(0,0))
        self.conv2 = nn.Conv2d(128, 256, kernel_size=(1,11), stride=(1,1), padding=(0,5))
        self.conv3 = nn.Conv2d(256, 512, kernel_size=(1,17), stride=(1,1), padding=(0,8))
        self.conv4 = nn.Conv2d(512, 512, kernel_size=(1,17), stride=(1,1), padding=(0,8))
        self.conv5 = nn.Conv2d(512, embedding_dim, kernel_size=(1,17), stride=(1,1), padding=(0,8))
        self.pool = nn.MaxPool2d(kernel_size=(1,3), stride=(1,2),padding=(0,1))

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.batchnorm1(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = F.relu(self.conv5(x))
        x = self.pool(x)
        x = x.squeeze(2)
        return x

class DavenetSmall(nn.Module):
    def __init__(self, input_dim, embedding_dim=1024):
        super(DavenetSmall, self).__init__()
        self.embedding_dim = embedding_dim
        self.batchnorm1 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(input_dim, 3), stride=(1,1), padding=(0,2))
        # self.conv1 = nn.Conv2d(1, 128, kernel_size=(40,1), stride=(1,1), padding=(0,0))
        self.conv2 = nn.Conv2d(64, 256, kernel_size=(1,3), stride=(1,1), padding=(0,1))
        self.conv3 = nn.Conv2d(256, 512, kernel_size=(1,3), stride=(1,1), padding=(0,2))
        self.pool = nn.MaxPool2d(kernel_size=(1,3), stride=(1,2),padding=(0,1))

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.batchnorm1(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = x.squeeze(2)
        return x

class SentenceRNN(nn.Module):
    def __init__(self, input_dim, embedding_dim,n_layers=3):
        super(SentenceRNN, self).__init__()
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.rnn = nn.LSTM(input_size=40, hidden_size=embedding_dim, num_layers=n_layers, batch_first=True, bidirectional=True)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)

        B = x.size(0)
        T = x.size(1)
        if torch.cuda.is_available():
          h0 = torch.zeros((2 * self.n_layers, B, self.embedding_dim))
          c0 = torch.zeros((2 * self.n_layers, B, self.embedding_dim))
        
        embed, _ = self.rnn(x, (h0, c0))
        print('embed.size(): ', embed.size())
        out = embed[:, :, :self.embedding_dim] + embed[:, :, self.embedding_dim:]
        return out



class CNN_RNN_ENCODER(nn.Module):
    def __init__(self,embedding_dim,n_layer):
        super(CNN_RNN_ENCODER,self).__init__()
        self.embedding_dim = embedding_dim
        self.Conv = nn.Conv1d(in_channels=49,out_channels=128,
                              kernel_size=5,stride=1,
                              padding=0)
        self.bnorm = nn.BatchNorm1d(128)
        self.rnn = nn.GRU(128, embedding_dim, n_layer, batch_first=True, dropout=0.5,
                        bidirectional=True)
        self.att = multi_attention(in_size = embedding_dim, hidden_size = 128, n_heads = 1)
    
    
    
    def forward(self, input, l):
            # input = input.transpose(2,1)
            x = self.Conv(input)
            x = self.bnorm(x)
            
            # l = [int((y-(self.Conv.kernel_size[0]-self.Conv.stride[0]))/self.Conv.stride[0]) for y in l]
            
            # create a packed_sequence object. The padding will be excluded from the update step
            # thereby training on the original sequence length only
            # x = torch.nn.utils.rnn.pack_padded_sequence(x.transpose(2,1), l, batch_first=True)
            x, hx = self.rnn(x.transpose(2,1))
            # unpack again as at the moment only rnn layers except packed_sequence objects
            # x, lens = nn.utils.rnn.pad_packed_sequence(x, batch_first = True)
            x = x[:, :, :self.embedding_dim] + x[:, :, self.embedding_dim:]
            x = self.att(x)

            return x
