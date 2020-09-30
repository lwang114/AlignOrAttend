import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as imagemodels
import torch.utils.model_zoo as model_zoo
import numpy as np
try:
  from .NegativeSquare import NegativeSquare
except:
  from NegativeSquare import NegativeSquare

class ResnetEncoder(nn.Module):
    def __init__(self, configs):
        super(ResnetEncoder, self).__init__()
        self.n_class = configs.get('n_class', 80)
        self.compute_softmax = configs.get('compute_softmax', False)
        self.activation = configs.get('softmax_activation', 'gaussian')
        self.embedding_dim = configs.get('embedding_dim', 512)
        self.codebook = None
        self.precision = configs.get('precision', 0.1)
        pretrained_model_file = configs.get('pretrained_model', None)
        codebook_file = configs.get('codebook_file', None)
        
        self.resnet = Resnet34(n_class=65) 
        if pretrained_model_file:
            self.resnet.load_state_dict(torch.load(pretrained_model_file))

        if self.activation == 'linear':
            self.clf = Linear(self.embedding_dim, self.n_class)
        elif self.activation == 'gaussian':
          if codebook_file:
            self.codebook = nn.Parameter(torch.FloatTensor(np.load(codebook_file)), requires_grad=False) # XXX
          else:
            self.codebook = nn.Parameter(torch.randn(self.n_class, self.embedding_dim), requires_grad=False) # XXX
          self.precision = nn.Parameter(self.precision*torch.ones(1, requires_grad=False))
          self.clf = NegativeSquare(self.codebook, self.precision)
        
    def forward(self, x, save_features=False):
        if len(x.size()) >= 5:
            B, L = x.size(0), x.size(1)
            x = torch.cat(torch.split(x, 1), dim=1).squeeze(0) 
            embed = self.resnet(x, save_features=True)[0] 
            embed = embed.view(B, L, -1)
        else:
            embed = self.resnet(x, save_features=True)[0]
        out = self.clf(embed)

        if self.compute_softmax:
          out = out.softmax(-1)
        if save_features:
            return embed, out
        else:
            return out

class LinearEncoder(nn.Module):
    def __init__(self, configs):
      super(LinearEncoder, self).__init__()
      self.n_class = configs.get('n_class', 80)
      self.compute_softmax = configs.get('compute_softmax', False)
      self.embedding_dim = configs.get('embedding_dim', 512)
      self.activation = configs.get('softmax_activation', 'gaussian')
      self.codebook = None
      self.precision = configs.get('precision', 0.1)
      codebook_file = configs.get('codebook_file', None)
      
      if self.activation == 'linear':
          self.clf = Linear(self.embedding_dim, self.n_class)
      elif self.activation == 'gaussian':
        if codebook_file:
          self.codebook = nn.Parameter(torch.FloatTensor(np.load(codebook_file)), requires_grad=False) # XXX
        else:
          self.codebook = nn.Parameter(torch.randn(self.n_class, self.embedding_dim), requires_grad=False) # XXX
        self.precision = nn.Parameter(self.precision*torch.ones(1, requires_grad=False))
        self.clf = NegativeSquare(self.codebook, self.precision)

    def forward(self, x, save_features=False):
      out = self.clf(x)
      if self.compute_softmax:
        out = out.softmax(-1)
        
      if save_features:
        return x, out
      else:
        return out
          
class Resnet18(imagemodels.ResNet):
    def __init__(self, embedding_dim=1024, pretrained=False):
        super(Resnet18, self).__init__(imagemodels.resnet.BasicBlock, [2, 2, 2, 2])
        if pretrained:
            self.load_state_dict(model_zoo.load_url(imagemodels.resnet.model_urls['resnet18']))
        self.avgpool = None
        self.fc = None
        self.embedder = nn.Conv2d(512, embedding_dim, kernel_size=1, stride=1, padding=0)
        self.embedding_dim = embedding_dim
        self.pretrained = pretrained

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.embedder(x)
        return x

class Resnet34(imagemodels.ResNet):
    def __init__(self, n_class=80,
                 pretrained=False):
        super(Resnet34, self).__init__(imagemodels.resnet.BasicBlock, [3, 4, 6, 3])
        self.fc = nn.Linear(512, n_class)
        self.codebook = None
        self.precision = None
        
        if pretrained:
            self.load_state_dict(model_zoo.load_url(imagemodels.resnet.model_urls['resnet34']))
            for child in self.conv1.children():
                for p in child.parameters():
                    p.requires_grad = False

            for child in self.layer1.children():
                for p in child.parameters():
                    p.requires_grad = False

            for child in self.layer2.children():
                for p in child.parameters():
                    p.requires_grad = False

            for child in self.layer3.children():
                for p in child.parameters():
                    p.requires_grad = False

            for child in self.layer4.children():
                for p in child.parameters():
                    p.requires_grad = False

            for child in list(self.avgpool.children()):
                for p in child.parameters():
                    p.requires_grad = False
            
            for p in self.fc.parameters():
              p.requires_grad = False


    def forward(self, x, save_features=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        embed = x.view(x.size(0), -1)
        x = self.fc(embed)
        if save_features:
            return embed, x
        else:
            return x   
            

class Resnet50(imagemodels.ResNet):
    def __init__(self, embedding_dim=1024, pretrained=False):
        super(Resnet50, self).__init__(imagemodels.resnet.Bottleneck, [3, 4, 6, 3])
        if pretrained:
            self.load_state_dict(model_zoo.load_url(imagemodels.resnet.model_urls['resnet50']))
        self.avgpool = None
        self.fc = None
        self.embedder = nn.Conv2d(2048, embedding_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.embedder(x)
        return x

class VGG16(nn.Module):
    def __init__(self, embedding_dim=1024, n_class=80, pretrained=False):
        super(VGG16, self).__init__()
        seed_model = imagemodels.__dict__['vgg16'](pretrained=pretrained).features
        seed_model = nn.Sequential(*list(seed_model.children())[:-1]) # remove final maxpool
        last_layer_index = len(list(seed_model.children()))
        seed_model.add_module(str(last_layer_index),
            nn.Conv2d(512, embedding_dim, kernel_size=(3,3), stride=(1,1), padding=(1,1)))
        seed_model.add_module(str(last_layer_index+1),
            nn.Linear(embedding_dim, n_class))
        self.image_model = seed_model

    def forward(self, x):
        x = self.image_model(x)
        return x
