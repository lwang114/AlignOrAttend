import torch
import torch.nn as nn

class NegativeSquare(nn.Module):
  def __init__(self, codebook, precision):
    super(NegativeSquare, self).__init__()
    self.codebook = nn.Parameter(codebook, requires_grad=True)
    self.precision = nn.Parameter(precision * torch.ones((1,)), requires_grad=False) # TODO Make this trainable                                                                                            

  def forward(self, x):
    """                                                                                                                                                                                                    
    Args:                                                                                                                                                                                                  
        x: B x T x D array of acoustic features                                                                                                                                                                                                                                                                                                                                                                       
    Returns:                                                                                                                                                                                                       score: B x T x K array of gaussian log posterior probabilities                                                                                                                                     
             [[[precision*(x[t] - mu[c[t]])**2 for k in range(K)] for t in range(T)] for b in range(B)]                                                                                                    
    """
    score = -(x.unsqueeze(-2) - self.codebook).pow(2).sum(-1)
    return self.precision * score
