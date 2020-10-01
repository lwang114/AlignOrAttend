import torch
import torch.nn as nn
import numpy as np

class LinearRetriever(nn.Module):
    def __init__(self, configs):
        super(LinearRetriever, self).__init__()
        self.embedding_dim = configs.get('embedding_dim', 200)
        self.input_dim = configs.get('input_dim', 512)
        self.simtype = configs.get('similarity_type', 'MISA')
        self.losstype = configs.get('loss_type', 'mml')
        self.fc = nn.Linear(self.input_dim, self.embedding_dim)

    def forward(self, x, y, x_mask, y_mask):
        """Calculate the dot product similarity matrix between data matrices x and y

        Args:
            x: B x L x D_in tensor   
            y: B x T x D tensor

        Returns:
            S: B x B similarity score matrix
        """
        B = x.size(0)
        Ls = x_mask.sum(-1).cpu().detach().numpy().astype(int)
        Ts = y_mask.sum(-1).cpu().detach().numpy().astype(int)

        S = torch.zeros((B, B), device=x.device)
        for i_x in range(B):
            for i_y in range(B):
                if Ls[i_x] == 0 or Ts[i_y] == 0:
                    continue
                S[i_x, i_y] = self.matchmap_similarity(x[i_x, :Ls[i_x]], y[i_y, :Ts[i_y]])
        return S

    def matchmap_similarity(self, x, y):
        if x.size(-1) != self.embedding_dim:
            x = self.fc(x)
        if y.size(-1) != self.embedding_dim:
            y = self.fc(y)

        matchmap = torch.mm(x, y.T)
        if self.simtype == 'MISA':
            return matchmap.max(0)[0].mean()
        elif self.simtype == 'SIMA':
            return matchmap.max(1)[0].mean()
        elif simtype == 'SISA':
            return matchmap.mean()

    def loss(self, x, y, x_mask, y_mask):
        S = self.forward(x, y, x_mask, y_mask)
        if self.losstype == 'mml':
            m = nn.LogSoftmax(-1)
            n = x.size(0)
            loss = -torch.sum(m(S).diag())-torch.sum(m(S.T).diag())
            loss = loss / n
        elif self.losstype == 'triplet':
            n = x.size(0)
            loss = torch.zeros(1, device=x.device, requires_grad=True)
            if n == 1:
                return loss
            for i in range(n):
                x_imp_ind = i
                y_imp_ind = i
                while x_imp_ind == i:
                    x_imp_ind = np.random.randint(0, n)
                while y_imp_ind == i:
                    y_imp_ind = np.random.randint(0, n)
                x2y_simdif = S[i, y_imp_ind] - S[i, i] + 1.
                if (x2y_simdif > 0).all():
                    loss = loss + x2y_simdif
                y2x_simdif = S[x_imp_ind, i] - S[i, i] + 1.
                if (y2x_simdif > 0).all():
                    loss = loss + y2x_simdif
            loss = loss / n
        else:
            raise NotImplementedError
        return loss


class DotProductRetriever(nn.Module):
    def __init__(self, configs):
        super(DotProductRetriever, self).__init__()
        self.embedding_dim = configs.get('embedding_dim', 512)
        self.simtype = configs.get('similarity_type', 'MISA')
        self.losstype = configs.get('loss_type', 'mml')

    def forward(self, x, y, x_mask, y_mask):
        """Calculate the dot product similarity matrix between data matrices x and y

        Args:
            x: B x L x D_in tensor   
            y: B x T x D tensor

        Returns:
            S: B x B similarity score matrix
        """
        B = x.size(0)
        Ls = x_mask.sum(-1).cpu().detach().numpy().astype(int)
        Ts = y_mask.sum(-1).cpu().detach().numpy().astype(int)
        S = torch.zeros((B, B), device=x.device)
        for i_x in range(B):
            for i_y in range(B):
                if Ls[i_x] == 0 or Ts[i_y] == 0:
                    continue
                S[i_x, i_y] = self.matchmap_similarity(x[i_x, :Ls[i_x]], y[i_y, :Ts[i_y]])
        return S

    def matchmap_similarity(self, x, y):
        print(x.size(), y.size())
        matchmap = torch.mm(x, y.T)
        if self.simtype == 'MISA':
            return matchmap.max(0)[0].mean() 
        elif self.simtype == 'SIMA':
            return matchmap.max(1)[0].mean()
        elif simtype == 'SISA':
            return matchmap.mean()

    def loss(self, x, y, x_mask, y_mask):
        S = self.forward(x, y, x_mask, y_mask)
        if self.losstype == 'mml':
            m = nn.LogSoftmax(-1)
            n = x.size(0)
            loss = -torch.sum(m(S).diag())-torch.sum(m(S.T).diag())
            loss = loss / n
        elif self.losstype == 'triplet':
            n = x.size(0)
            loss = torch.zeros(1, device=x.device, requires_grad=True)
            for i in range(n):
                x_imp_ind = i
                y_imp_ind = i
                while x_imp_ind == i:
                    x_imp_ind = np.random.randint(0, n)
                while y_imp_ind == i:
                    y_imp_ind = np.random.randint(0, n)
                x2y_simdif = S[i, y_imp_ind] - S[i, i] + 1.
                if (x2y_simdif > 0).all():
                    loss = loss + x2y_simdif
                y2x_simdif = S[x_imp_ind, i] - S[i, i] + 1.
                if (y2x_simdif > 0).all():
                    loss = loss + y2x_simdif
            loss = loss / n
        else:
            raise NotImplementedError
        return loss
