import json
import numpy as np
import os
from PIL import Image
import scipy.signal
import torch
import torch.nn.functional
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class ImagePhoneCaptionDataset(Dataset):
  def __init__(self, phone_feat_file, image_feat_file, phone2idx_file, split_file=None, feat_conf={}):
    # Inputs:
    # ------  
    #   image_feat_file: .npz file with the format {'arr_1': y_1, ..., 'arr_n': y_n}, where y_i is an N x ndim array  
    #   phone_feat_file: .txt file with each line as a phone caption
    #
    # Outputs:
    # -------
    #   None
    self.max_nregions = feat_conf.get('max_num_regions', 5)
    self.max_nphones = feat_conf.get('max_num_phones', 100)
    self.image_first = feat_conf.get('image_first', True)
    self.split_file = split_file
    with open(phone2idx_file, 'r') as f:
      self.phone2idx = json.load(f)
    self.phone_feats = []
    self.word_boundaries = []
    n_types = len(self.phone2idx)

    print(phone_feat_file, image_feat_file)
    image_feat_npz = np.load(image_feat_file)
    self.image_feats = [image_feat_npz[k] for k in sorted(image_feat_npz, key=lambda x:int(x.split('_')[-1]))[:30]] # XXX 
    if self.split_file:
      with open(self.split_file, 'r') as f:
          self.selected_indices = [i for i, line in enumerate(f)][:30] # if int(line)] # XXX
    else:
      self.selected_indices = list(range(len(self.image_feats)))

    # Load the phone captions
    with open(phone_feat_file, 'r') as f:
      i = 0
      for line in f:
        a_sent = line.strip().split()
        if len(a_sent) == 0:
          print('Warning: empty caption', i)
        if i > 29: # XXX
          break
        i += 1

        a_feat = np.zeros((self.max_nphones, n_types))
        a_boundary = np.zeros((self.max_nphones+1,))
        start = 0
        a_boundary[start] = 1.
        for word in a_sent:
          phns = word.split(',')
          word_len = len(phns)
          a_boundary[min(start+word_len, self.max_nphones)] = 1.
          for t, phn in enumerate(phns):
            if start + t >= self.max_nphones:
              break
            phn_idx = self.phone2idx[phn.lower()]
            a_feat[start + t, phn_idx] = 1.
          start = start + word_len
            
        self.phone_feats.append(a_feat)
        self.word_boundaries.append(a_boundary)
        
    print(len(self.phone_feats), len(self.image_feats))
    assert len(self.phone_feats) == len(self.image_feats) 
    print('---- Dataset Summary ----')
    print('Number of examples: ', len(self.phone_feats))
    print('Number of phone types: ', n_types)

  def __len__(self):
    return len(self.selected_indices)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      new_idx = [self.selected_indices[i] for i in idx.tolist()]
    else:
      new_idx = self.selected_indices[idx]
      
    image_feat = self.image_feats[new_idx]
    nregions = min(len(image_feat), self.max_nregions)
    region_mask = torch.zeros((self.max_nregions,))
    region_mask[:nregions+1] = 1.

    image_feat = self.convert_to_fixed_length(image_feat)
    if self.image_first:
        return torch.FloatTensor(image_feat), torch.FloatTensor(self.phone_feats[new_idx]), region_mask, torch.FloatTensor(self.word_boundaries[new_idx])
    else:
        return torch.FloatTensor(self.phone_feats[new_idx]), torch.FloatTensor(image_feat), torch.FloatTensor(self.word_boundaries[new_idx]), region_mask

  def convert_to_fixed_length(self, image_feat):
    N = image_feat.shape[0]
    pad = abs(self.max_nregions - N)
    if N < self.max_nregions:
      image_feat = np.pad(image_feat, ((0, pad), (0, 0)), 'constant', constant_values=(0))
    elif N > self.max_nregions:
      image_feat = image_feat[:-pad, :]
    return image_feat
