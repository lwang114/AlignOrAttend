import torch
import torch.nn as nn

class LinearRetriever(nn.Module):
    def __init__(self, configs):
        super(LinearRetriever, self).__init__()
        self.embedding_dim = configs.get('embedding_dim', 200)
        self.input_dim = configs.get('input_dim', 512)
        self.simtype = configs.get('similarity_type', 'MISA')
        self.losstype = configs.get('loss_type', 'mml')
        self.fc = nn.Linear(self.input_dim, self.embedding_dim)

    def forward(self, x, y):
        """Calculate the dot product similarity matrix between data matrices x and y

        Args:
            x: B x L x D_in tensor   
            y: B x T x D tensor

        Returns:
            S: B x B similarity score matrix
        """
        B = x.size(0)

        S = torch.zeros((B, B), device=x.device)
        for i_x in range(B):
            for i_y in range(B):
                S[i_x, i_y] = self.matchmap_similarity(x[i_x], y[i_y]) # TODO Apply mask
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

    def loss(self, x, y):
        S = self.forward(x, y)
        if self.losstype == 'mml':
            m = nn.LogSoftmax(-1)
            n = x.size(0)
            loss = -torch.sum(m(S).diag())-torch.sum(m(S.T).diag())
            loss = loss / 2
        else:
            raise NotImplementedError
        return loss


class DotProductRetriever(nn.Module):
    def __init__(self, configs):
        super(DotProductRetriever, self).__init__()
        self.embedding_dim = configs.get('embedding_dim', 512)
        self.simtype = configs.get('similarity_type', 'MISA')
        self.losstype = configs.get('loss_type', 'mml')

    def forward(self, x, y):
        """Calculate the dot product similarity matrix between data matrices x and y

        Args:
            x: B x L x D_in tensor   
            y: B x T x D tensor

        Returns:
            S: B x B similarity score matrix
        """
        B = x.size(0)
        S = torch.zeros((B, B), device=x.device)
        for i_x in range(B):
            for i_y in range(B):
                S[i_x, i_y] = self.matchmap_similarity(x[i_x], y[i_y]) # TODO Apply mask
        return S

    def matchmap_similarity(self, x, y):
        matchmap = torch.mm(x, y.T)
        if self.simtype == 'MISA':
            return matchmap.max(0)[0].mean()
        elif self.simtype == 'SIMA':
            return matchmap.max(1)[0].mean()
        elif simtype == 'SISA':
            return matchmap.mean()

    def loss(self, x, y):
        S = self.forward(x, y)
        if self.losstype == 'mml':
            m = nn.LogSoftmax(-1)
            n = x.size(0)
            loss = -torch.sum(m(S).diag())-torch.sum(m(S.T).diag())
            loss = loss / 2
        else:
            raise NotImplementedError
        return loss


