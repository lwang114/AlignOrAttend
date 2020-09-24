import torch
import torch.nn as nn
import numpy as np

class NoopSegmenter(nn.Module):
  """Dummy segmenter that performs no segmentation"""
  def __init__(self, configs={}):
    super(NoopSegmenter, self).__init__()
    self.max_nframes = configs.get('max_nframes', 500)
    self.max_nsegments = configs.get('max_nsegments', 50)
    
  def forward(self, x, in_boundary=[]): 
    B = x.size(0)
    if in_boundary is None:
      in_boundary = torch.ones((x.size(0), x.size(1)+1), device=x.device)

    mask = torch.zeros((B, self.max_nsegments, self.max_nframes))
    for b in range(B):
      segment_times = np.nonzero(in_boundary[b].cpu().detach().numpy())[0]
      if segment_times[0] != 0:
        segment_times = np.append(0, segment_times)

      for t, (start, end) in enumerate(zip(segment_times[:-1], segment_times[1:])):
        if t >= self.max_nsegments or end > self.max_nframes:
          break
        mask[b, t, start:end] = 1. / (end - start) if end - start > 0 else 1.

    mask = torch.FloatTensor(mask).to(x.device)
    x = torch.matmul(mask, x)
    return x, mask.sum(-1), in_boundary
