import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as imagemodels
import torch.utils.model_zoo as model_zoo

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
    def __init__(self, n_class=80, pretrained=False):
        super(Resnet34, self).__init__(imagemodels.resnet.BasicBlock, [3, 4, 6, 3])
        self.fc = nn.Linear(512, n_class)
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


    def forward(self, x):
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
        return x
    
    # def cluster(self, x): # TODO

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

class NoOpEncoder(nn.Module):
    def __init__(self, embedding_dim=1024, pretrained=False):
      self.embedding_dim=embedding_dim
        
    def forward(self, x):
      return x     
