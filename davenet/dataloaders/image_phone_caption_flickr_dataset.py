import json
import librosa
import numpy as np
import os
from PIL import Image
import scipy.signal
import torch
import torch.nn.functional
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class ImagePhoneCaptionFlickrDataset(Dataset):
  def __init__(self, data_dir, split='train',
               max_nregions=5,
               image_feat_type='res34'):
    # Inputs:
    # ------  
    #   image_feat_file: .npz file with the format {'arr_1': y_1, ..., 'arr_n': y_n}, where y_i is an N x ndim array  
    #   phone_feat_file: .txt file with each line as a phone caption
    #
    # Outputs:
    # -------
    #   None
    self.split = split
    self.max_nregions = max_nregions
    self.max_nphones = 100
    self.base_root = data_dir
    self.file_root = os.path.join(data_dir)
    self.phone2idx = self.load_p2i(self.file_root)
    # self.filenames = self.load_filenames(self.file_root, split)
    image_path = os.path.join(self.base_root, 'flickr30k_{}.npz'.format(image_feat_type))
    image_feats_npz = np.load(image_path)
    with open(os.path.join(self.base_root, 'flickr8k_test.txt')) as f:
      test_keys = f.read().strip().split('\n')
      test_keys = ['_'.join(k.split('_')[:2]) for k in test_keys]

    if self.split == 'train':  
      self.filenames = [k for k in sorted(image_feats_npz, key=lambda x:int(x.split('_')[-1])) if not k.split('.')[0] in test_keys] # XXX
      keep = [ex for ex, k in enumerate(sorted(image_feats_npz, key=lambda x:int(x.split('_')[-1]))) if not k.split('.')[0] in test_keys] # XXX
    else:
      self.filenames = [k for k in sorted(image_feats_npz, key=lambda x:int(x.split('_')[-1])) if k.split('.')[0] in test_keys and k.split('_')[2] == '1'] # XXX
      keep = [ex for ex, k in enumerate(sorted(image_feats_npz, key=lambda x:int(x.split('_')[-1]))) if k.split('.')[0] in test_keys and k.split('_')[2] == '1'] # XXX
    
    self.image_feats = [image_feats_npz[fn] for fn in self.filenames]
    print('Number of images={}'.format(len(self.filenames)))
    self.load_phone(data_dir, keep)

  def load_p2i(self,root_path):
    path = os.path.join(root_path, 'flickr_phone2id.json')
    if os.path.isfile(path):
      with open(path,'r') as f:
        data = json.load(f)
    else:
      data = {}
      with open(path, 'w') as f_p2i,\
           open(os.path.join(root_path, 'flickr30k_phone_captions_filtered.txt'), 'r') as f_c:
        for line in f_c:
          a_sent = line.strip().split()
          for phn in a_sent:
            if not phn in data:
              data[phn.lower()] = len(data)
        json.dump(data, f_p2i, indent=4, sort_keys=True)
    return data

  def load_filenames(self,root_path,split):
    if split == 'train':
        path = os.path.join(root_path,'mscoco_filenams_train.json')
    else:
        path = os.path.join(root_path,'mscoco_filenams_val_1000.json')
    with open(path,'r') as f:
      data = json.load(f)
    return data

  def convert_to_fixed_length(self, image_feat):
    N = image_feat.shape[0]
    while N < self.max_nregions:
      image_feat = image_feat.repeat(2,0)
      N = image_feat.shape[0]
    if N > self.max_nregions:
      image_feat = image_feat[:self.max_nregions, :]
    return image_feat

  def load_phone(self,data_dir, keep):
    n_types = len(self.phone2idx)
    path = os.path.join(data_dir, 'flickr30k_phone_captions_filtered.txt')
    with open(path,'r') as f:
      data = f.read().splitlines()
    
    with open(path, 'r') as f:
      self.phone_feats = [] 
      self.nphones = []
      i = 0
      for line in f:
        if not i in keep:
          i += 1
          continue
        a_sent = line.strip().split()
        phone_num = len(a_sent)
        if len(a_sent) == 0:
          phone_num = 1
          # print('Empty caption', i)
        # if len(self.phone_feats) == 30: # XXX
        #   break
        i += 1
        a_feat = np.zeros((n_types, self.max_nphones))
        for phn in a_sent:
          for t, phn in enumerate(a_sent):
            if t >= self.max_nphones:
              break
            a_feat[self.phone2idx[phn.lower()], t] = 1.
        self.phone_feats.append(a_feat)
        self.nphones.append(min(phone_num, self.max_nphones))    
    print('Number of captions={}'.format(len(self.phone_feats)))

  def __getitem__(self, idx):
    # key = self.filenames[idx]
    # load image
    image_feat = self.image_feats[idx]
    image_feat = self.convert_to_fixed_length(image_feat)

    return torch.FloatTensor(self.phone_feats[idx]), torch.FloatTensor(image_feat), self.nphones[idx], self.max_nregions

  def __len__(self):
    return len(self.filenames)
