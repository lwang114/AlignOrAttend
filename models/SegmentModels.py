import torch
import torch.nn as nn
import numpy as np

class NoopSegmenter(nn.Module):
  """Dummy segmenter that performs no segmentation"""
  def __init__(self, configs={}):
    super(NoopSegmenter, self).__init__()
    self.max_nframes = configs.get('max_nframes', 500)
    self.max_nsegments = configs.get('max_nsegments', 50)
    self.n_intervals = configs.get('n_intervals', 1)

  def forward(self, x, in_boundary=None, is_embed=False): 
    B = x.size(0)
    L = x.size(1)
    if in_boundary is None:
      in_boundary = torch.ones((B, L+1), device=x.device)

    n_intervals = self.n_intervals if is_embed else 1 
    mask = torch.zeros((B, self.max_nsegments*n_intervals, self.max_nframes))
    for b in range(B):
      segment_times = np.nonzero(in_boundary[b].cpu().detach().numpy())[0]
      if segment_times[0] != 0:
        segment_times = np.append(0, segment_times)

      for t, (start, end) in enumerate(zip(segment_times[:-1], segment_times[1:])):
        if t >= self.max_nsegments or end > self.max_nframes:
          break
        
        interval_len = round((end - start) / n_intervals)
        for i_itvl in range(n_intervals):
            if interval_len == 0:
                mask[b, t, ]
            start_interval = int(start + i_itvl * interval_len)
            end_interval = int(min(start + (i_itvl + 1) * interval_len, end))
            cur_interval_len = end_interval - start_interval
            mask[b, t, start_interval:end_interval] = 1. / cur_interval_len if cur_interval_len > 0 else 0.

    mask = torch.FloatTensor(mask).to(x.device)
    x = torch.matmul(mask, x).view(B, self.max_nsegments, -1)
    mask = mask.sum(-1)[:, ::n_intervals]
    return x, mask, in_boundary

class FixedTextSegmenter(nn.Module):
  """Segmenter that performs no segmentation for phone sequence"""
  def __init__(self, configs={}):
    super(FixedTextSegmenter, self).__init__()
    self.max_nsegments = configs.get('max_nsegments', 50)
    self.max_vocab_size = configs.get('max_vocab_size', 3001)
    self.word2idx = {'UNK':0}
    
  def forward(self, x, in_boundary=None, is_embed=False):
    B = x.size(0)
    L = x.size(1)

    if in_boundary is None:
      in_boundary = torch.ones((B, L+1), device=x.device)

    segment_times = np.nonzero(in_boundary.cpu().detach().numpy())[0]
    mask = torch.zeros((B, self.max_nsegments), device=x.device)
    output = []

    for b in range(B):
      sent = torch.zeros((self.max_nsegments, self.max_vocab_size), device=x.device)
      for t, (start, end) in enumerate(zip(segment_times[:-1], segment_times[1:])):
        if t >= self.max_nsegments:
          break
        mask[b, t] = 1.
        phns = np.argmax(x[b, start:end].cpu().detach().numpy(), axis=-1)
        word = ' '.join(str(phn) for phn in phns)
        if not word in self.word2idx and len(self.word2idx) < self.max_vocab_size:
          self.word2idx[word] = len(self.word2idx)
          sent[t, self.word2idx[word]] = 1.
        else:
          sent[t, 0] = 1.
      output.append(sent)
    
    if is_embed:
        return x, mask, in_boundary
    else:
        return torch.stack(output), mask, in_boundary
