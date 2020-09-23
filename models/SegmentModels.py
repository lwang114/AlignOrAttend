import torch
import numpy as np

class NoopSegmenter(nn.Module):
  """Dummy segmenter that performs no segmentation"""
  def __init__(self, model_conf=None):
    super(NoopSegmenter, self).__init__()
    self.model_conf = model_conf

  def forward(self, x, landmark=None): 
    return landmark
    
  def embed(self, x, landmark): 
    B = x.size(0)

    for b in range(B):
      segment_times = np.nonzero(landmark[b].cpu().detach().numpy())[0]
      if segment_times[0] != 0:
        segment_times = np.append(0, segment_times)
      for t, (start, end) in enumerate(zip(segment_times[:-1], segment_times[1:])):
        mask[b, t, start:end] = 1. / max(end - start, 1)

    mask = torch.FloatTensor(mask).to(x.device)
    x = torch.matmul(mask, x)
    return x, mask.sum(-1)
