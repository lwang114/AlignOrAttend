import torch
import torch.nn as nn
import torch.nn.functional as F
from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
import numpy as np
from .NegativeSquare import NegativeSquare

class Transformer(nn.Module):
  def __init__(self, n_class,
               pretrained_model_file=None,
               model_conf_file=None,
               feat_conf_file=None):
    super(Transformer, self).__init__()
    model_conf, feat_conf = {}, {}
    if model_conf_file and feat_conf_file:
      with open(model_conf_file, 'r') as fm,\
           open(feat_conf_file, 'r') as ff:
        model_conf = yaml.safe_load(fm)
        feat_conf = yaml.safe_load(ff)
      
    idim = feat_conf.get('ndim', 83)
    num_blocks = model_conf.get('elayers', 12)
    input_layer = model_conf.get('input_layer', 'conv2d')
    linear_units = model_conf.get('eunits', 2048)
    attention_dim = model_conf.get('adim', 256)
    attention_heads = model_conf.get('aheads', 4)
    embedding_dim= model_conf.get('adim', 256)
    
    self.encoder = Encoder(idim=idim,
                           attention_dim=attention_dim,
                           attention_heads=attention_heads,
                           linear_units=linear_units,
                           input_layer=input_layer ,
                           num_blocks=num_blocks)
    # for k in self.encoder.state_dict():
    #   print(k, self.encoder.state_dict()[k].size())
    
    if pretrained_model_file:
      self.encoder.load_state_dict(torch.load(pretrained_model_file))
    self.fc = nn.Linear(embedding_dim, n_class) 

  def forward(self, x, save_features=False):
    if x.dim() < 3:
      x = x.unsqueeze(0)

    masks = make_non_pad_mask([x.size(1)]*x.size(0)).to(x.device).unsqueeze(-2)
    embed, _ = self.encoder(x, masks)
    output = self.fc(embed)
    if save_features:
      return embed, output
    else:
      return output

class BLSTMEncoder(nn.Module):
  def __init__(self, configs):
    super(BLSTMEncoder, self).__init__()
    self.n_class = configs.get('n_class', 1000)
    self.activation = configs.get('softmax_activation', 'gaussian')
    self.embedding_dim = configs.get('embedding_dim', 200)
    self.codebook = None
    self.precision = configs.get('precision', 0.1)
    pretrained_model_file = configs.get('pretrained_model', None)
    
    self.lstm = BLSTM3(49, self.embedding_dim) # XXX Assume the lstm is pretrained on phone clf task
    if pretrained_model_file:
      self.lstm.load_state_dict(torch.load(pretrained_model_file))
      
    if self.activation == 'linear':
      self.clf = Linear(2*self.embedding_dim, self.n_class)
    elif self.activation == 'gaussian':
      self.codebook = nn.Parameter(torch.randn(self.n_class, 2*self.embedding_dim),
                                   requires_grad=True)
      self.clf = NegativeSquare(self.codebook, self.precision)

  def forward(self, x, save_features=False):
    embed = self.lstm(x, save_features=True)[0]
    out = self.clf(embed)
    if save_features:
      return embed, out
    else:
      return out
    
class BLSTM2(nn.Module):
  def __init__(self, n_class, embedding_dim=100, n_layers=1, batch_first=True):
    super(BLSTM2, self).__init__()
    self.embedding_dim = embedding_dim
    self.n_layers = n_layers
    self.n_class = n_class
    self.batch_first = batch_first
    self.rnn = nn.LSTM(input_size=40,
                       hidden_size=embedding_dim,
                       num_layers=n_layers,
                       batch_first=batch_first,
                       bidirectional=True)
    self.fc = nn.Linear(2 * embedding_dim, n_class)
    self.softmax = nn.Softmax(dim=1)
    
  def forward(self, x, save_features=False):
    if x.dim() < 3:
      x = x.unsqueeze(0)

    B = x.size(0)
    T = x.size(1)
    h0 = torch.zeros((2 * self.n_layers, B, self.embedding_dim), device=x.device)
    c0 = torch.zeros((2 * self.n_layers, B, self.embedding_dim), device=x.device)
    embed, _ = self.rnn(x, (h0, c0))
    outputs = []
    for b in range(B):
      # out = self.softmax(self.fc(embed[b]))
      out = self.fc(embed[b])
      outputs.append(out)

    if save_features:
      return embed, torch.stack(outputs, dim=(0 if self.batch_first else 1))
    else:
      return torch.stack(outputs, dim=(0 if self.batch_first else 1))

  
class BLSTM3(nn.Module):
  def __init__(self, n_class,
               embedding_dim=100,
               n_layers=2,
               layer1_pretrain_file=None,
               batch_first=True,
               return_empty=False):
    super(BLSTM3, self).__init__()
    self.embedding_dim = embedding_dim
    self.n_layers = n_layers
    self.n_class = n_class
    self.batch_first = batch_first
    self.return_empty = return_empty

    self.rnn1 = BLSTM2(n_class, embedding_dim, batch_first=batch_first)
    self.rnn2 = nn.LSTM(input_size=2*embedding_dim,
                        hidden_size=embedding_dim,
                        num_layers=n_layers,
                        batch_first=batch_first,
                        bidirectional=True)
    self.fc = nn.Linear(2 * embedding_dim, n_class)

    if layer1_pretrain_file:
      self.rnn1.load_state_dict(torch.load(layer1_pretrain_file))      

    for child in self.rnn1.children():
      for p in child.parameters():
        p.requires_grad = False

    for p in self.rnn2.parameters():
      p.requires_grad = False
    
    for p in self.fc.parameters():
      p.requires_grad = False

  def forward(self, x,
              save_features=False):
    x, _ = self.rnn1(x, save_features=True) 
    B = x.size(0)
    T = x.size(1)
    h0 = torch.zeros((2 * self.n_layers, B, self.embedding_dim), device=x.device)
    c0 = torch.zeros((2 * self.n_layers, B, self.embedding_dim), device=x.device)
    embed, _ = self.rnn2(x)
    outputs = [self.fc(embed[b]) for b in range(B)]
    outputs = torch.stack(outputs, dim=(0 if self.batch_first else 1))
          
    if save_features:
      if self.return_empty:
        return embed, outputs
      else: # Assume empty index is 0
        return embed, outputs[:, :, 1:]
    else:
      if self.return_empty:
        return outputs
      else: # Assume empty index is 0
        return outputs[:, :, 1:]

class TDNN3(nn.Module):
  def __init__(self, n_class, embedding_dim=128):
    super(TDNN3, self).__init__()
    self.embedding_dim = embedding_dim
    self.batchnorm1 = nn.BatchNorm2d(1)
    # self.conv1 = nn.Conv2d(1, 128, kernel_size=(40, 3), stride=(1, 1), padding=(0, 1))
    # XXX
    '''
    self.conv1 = nn.Conv2d(1, 32, kernel_size=(40, 3), stride=(1, 1), padding=(0, 1))
    self.conv2 = nn.Conv2d(32, 64, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2))
    self.conv3 = nn.Conv2d(64, embedding_dim, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2))
    self.pool = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)) 
    # XXX
    
    self.avgpool = nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=(0, 0))
    
    self.fc = nn.Linear(embedding_dim, n_class) 
    '''
    '''
    self.td1 = TDNN(input_dim=input_dim, 
                    output_dim=10,
                    context_size=1)
    self.td2 = TDNN(input_dim=input_dim, 
                    output_dim=5,
                    context_size=2)
    '''
    self.fc1 = nn.Linear(600, 1000)
    self.fc2 = nn.Linear(1000, n_class)

  def forward(self, x, save_features=False):
    '''
    if x.dim() == 3:
      x = x.unsqueeze(1)
    '''
    x = x.view(x.size(0), -1)
    '''
    x = self.batchnorm1(x)
    x = F.relu(self.conv1(x))
    # XXX
    x = self.pool(x)
    x = F.relu(self.conv2(x))
    x = self.pool(x)
    x = F.relu(self.conv3(x))
    '''
    '''
    x = self.avgpool(x)
    embed = x.squeeze()'''
    '''
    embed = torch.mean(x, -1).squeeze()
    out = self.fc(embed)
    '''
    # print(self.fc1.weight.dtype, self.fc1.bias.dtype)
    embed = F.relu(self.fc1(x))
    out = self.fc2(embed)

    if save_features:
      return embed, out
    else:
      return out
