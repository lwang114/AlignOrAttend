import numpy as np
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sphfile import SPHFile
import scipy.io.wavfile as wavfile
import librosa
import kaldiio
from PIL import Image

EPS = 1e-9
# This function is from DAVEnet (https://github.com/dharwath/DAVEnet-pytorch)
def preemphasis(signal,coeff=0.97):
    """perform preemphasis on the input signal.
    
    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is none, default 0.97.
    :returns: the filtered signal.
    """    
    return np.append(signal[0],signal[1:]-coeff*signal[:-1])

class ImageAudioCaptionDataset(Dataset):
  def __init__(self, audio_root_path, image_root_path, segment_file, bbox_file, keep_index_file=None, configs={}):
    self.configs = configs
    self.max_nregions = configs.get('max_num_regions', 5)
    self.max_nphones = configs.get('max_num_phones', 50)
    self.max_nframes = configs.get('max_num_frames', 500)
    self.segment_level = configs.get('segment_level', 'word')
    self.image_first = configs.get('image_first', True)
    self.audio_keys = []
    self.segment_file = segment_file
    self.bbox_file = bbox_file
    self.audio_root_path = audio_root_path
    self.segmentations = []
    self.image_keys = []
    self.image_root_path = image_root_path 
    self.bboxes = []

    if self.audio_root_path.split('.')[-1] == 'json': # Assume kaldi format if audio_root_path is a json file
        with open(audio_root_path, 'r') as f:
            audio_feat_dict = json.load(f)['utts']
    
    # Load phone segments
    with open(segment_file, 'r') as f:
      if segment_file.split('.')[-1] == 'json':
        segment_dicts = json.load(f)
        for k in sorted(segment_dicts, key=lambda x:int(x.split('_')[-1])):
          audio_file_prefix = '_'.join('_'.join(word_segment_dict[1].split('_')[:-1]) for word_segment_dict in segment_dicts[k]['data_ids'])
          if self.audio_root_path.split('.')[-1] == 'json':
              k_expanded = '{}_{:06d}'.format('_'.join(k.split('_')[:-1]), int(k.split('_')[-1]))
              self.audio_keys.append(audio_feat_dict[k_expanded]['input'][0]['feat'])
              
          else:
              self.audio_keys.append(audio_file_prefix)
          segmentation = []
          cur_start = 0
          for word_segment_dict in segment_dicts[k]['data_ids']:
              dur = 0.
              for phone_segment in word_segment_dict[2]:
                  if self.segment_level == 'phone':
                    dur = phone_segment[2] - phone_segment[1]
                    segmentation.append([cur_start, cur_start+dur])
                    cur_start += dur
                  elif self.segment_level == 'word':
                    dur += phone_segment[2] - phone_segment[1]
                  else:
                    raise ValueError('Unknown segment level')

              if self.segment_level == 'word':
                segmentation.append([cur_start, cur_start+dur])

          self.segmentations.append(segmentation)
          # if len(self.segmentations) > 29: # XXX
          #   break
      else:
        for line in f:
          k, phn, start, end = line.strip().split()
          if len(self.audio_keys) == 0:
            self.audio_keys.append(k)
            self.segmentations.append([[start, end]])
          elif k != self.audio_keys[-1]:
            self.audio_keys.append(k)
            self.segmentations.append([[start, end]])
          else:
            self.segmentations[-1].append([start, end])
          # if len(self.segmentations) > 29: # XXX
          #   break

    if bbox_file.split('.')[-1] == 'txt' and segment_file.split('.')[-1] == 'json':
        with open(segment_file, 'r') as fs:
            image_dicts = json.load(fs) 

        bbox_dict = {} 
        with open(bbox_file, 'r') as fb:
            for line in fb:
                k, c, x, y, w, h = line.strip().split()
                bbox_dict[k] = [x, y, w, h]

        for k in sorted(image_dicts, key=lambda x:int(x.split('_')[-1])):
            image_list = [image_dict[0] for image_dict in image_dicts[k]['data_ids']]                
            image_file_prefix = ':'.join(image_list)
            self.image_keys.append(image_file_prefix)
            self.bboxes.append([bbox_dict[img_id] for img_id in image_list])
            # if len(self.bboxes) > 30: # XXX
            #   break
    elif bbox_file.split('.')[-1] == 'txt':    
        with open(bbox_file, 'r') as f:
          for line in f:
              k, c, x, y, w, h = line.strip().split() 
              # XXX self.image_keys.append('_'.join(k.split('_')[:-1]))
              if len(self.image_keys) == 0:
                  self.image_keys = [k]
        
              elif k != self.image_keys[-1]:
                  self.image_keys.append(k)
                  self.bboxes.append([[x, y, w, h]]) 
              else:
                  self.bboxes[-1].append([x, y, w, h])
    elif bbox_file.split('.')[-1] == 'npz': # If bbox file is npz format, assume the features are already extracted 
        self.bboxes = np.load(bbox_file) 
        self.image_keys = sorted(self.bboxes, key=lambda x:int(x.split('_')[-1])) # XXX

    self.keep_indices = None
    if keep_index_file:
        with open(keep_index_file, 'r') as f:
            self.keep_indices = [i for i, line in enumerate(f) if int(line)] # XXX
    else:
        self.keep_indices = list(range(len(self.audio_keys)))
    print('Number of images = {}, number of captions = {}'.format(len(self.image_keys), len(self.audio_keys)))
    print('Keep {} image-caption pairs'.format(len(self.keep_indices)))

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()

    mfcc, phone_boundary = self.load_audio(idx)
    regions, region_mask = self.load_image(idx)
    if self.image_first:
        return regions, mfcc, region_mask, phone_boundary
    else:
        return mfcc, regions, phone_boundary, region_mask
  
  def load_audio(self, idx):
    idx = self.keep_indices[idx]
    # Extract segment-level acoustic features
    self.n_mfcc = self.configs.get('n_mfcc', 40)
    self.coeff = self.configs.get('coeff', 0.97)
    self.dct_type = self.configs.get('dct_type', 3)
    self.skip_ms = self.configs.get('skip_size', 10)
    self.window_ms = self.configs.get('window_len', 25)

    phone_boundary = np.zeros(self.max_nframes+1)
    for i_s, segment in enumerate(self.segmentations[idx]):
      start_ms, end_ms = segment
      start_frame, end_frame = int(float(start_ms) / 10), int(float(end_ms) / 10)
      if end_frame > self.max_nframes:
        break
      phone_boundary[start_frame] = 1.
      phone_boundary[end_frame] = 1.

    if self.audio_root_path.split('.')[-1] == 'json': # Assume kaldi format if audio_root_path is a json file
        mfcc = kaldiio.load_mat(self.audio_keys[idx])
        mfcc = self.convert_to_fixed_length(mfcc.T).T
    else:
        audio_filename = '{}.wav'.format(self.audio_keys[idx])
        try:
            sr, y_wav = wavfile.read('{}/{}'.format(self.audio_root_path, audio_filename))
        except:
            if audio_filename.split('.')[-1] == 'wav':
                audio_filename_sph = '.'.join(audio_filename.split('.')[:-1]+['WAV'])
                sph = SPHFile(self.audio_root_path + audio_filename_sph)
                sph.write_wav(self.audio_root_path + audio_filename_sph)
            sr, y_wav = wavfile.read(self.audio_root_path + audio_filename)
    
        y_wav = preemphasis(y_wav, self.coeff) 
        n_fft = int(self.window_ms * sr / 1000)
        hop_length = int(self.skip_ms * sr / 1000)
        mfcc = librosa.feature.mfcc(y_wav, sr=sr, n_mfcc=self.n_mfcc, dct_type=self.dct_type, n_fft=n_fft, hop_length=hop_length)
        mfcc -= np.mean(mfcc)
        mfcc /= max(np.sqrt(np.var(mfcc)), EPS)
        nframes = min(mfcc.shape[1], self.max_nframes)
        mfcc = self.convert_to_fixed_length(mfcc)
        mfcc = mfcc.T
    
    mfcc = torch.FloatTensor(mfcc)
    if phone_boundary.sum() == 0:
        print('Warning: Caption {} with id {} is empty'.format(idx, self.audio_keys[idx]))
        phone_boundary[0] = 1.
    phone_boundary = torch.FloatTensor(phone_boundary)

    return mfcc, phone_boundary

  def load_image(self, idx):
    idx = self.keep_indices[idx]
    self.width = self.configs.get('width', 224)
    self.height = self.configs.get('height', 224)
    RGB_mean = self.configs.get('RGB_mean', [0.485, 0.456, 0.406])
    RGB_std = self.configs.get('RGB_std', [0.229, 0.224, 0.225])
    self.transform = self.configs.get('transform', transforms.Compose([transforms.Resize(256),
                                                                  transforms.CenterCrop(224),
                                                                  transforms.ToTensor(),
                                                                  transforms.Normalize(mean=RGB_mean, std=RGB_std)])) 
    # Extract visual features 
    regions = []
    region_mask = np.zeros(self.max_nregions+1)
    if self.bbox_file.split('.')[-1] == 'npz':
        regions = torch.FloatTensor(self.bboxes[self.image_keys[idx]][:self.max_nregions])
        if len(regions) < self.max_nregions:
            regions = torch.cat((regions, torch.zeros((self.max_nregions-len(regions), regions.size(-1)))))
        # TODO Handle empty region
        region_mask[:min(len(regions), self.max_nregions)+1] = 1.
    else: 
        for i_b, bbox in enumerate(self.bboxes[idx]):
          if i_b > self.max_nregions:
            break
          x, y, w, h = bbox 
          x, y, w, h = int(x), int(y), np.maximum(int(w), 1), np.maximum(int(h), 1)
          if ':' in self.image_keys[idx]:
              image = Image.open('{}/{}.jpg'.format(self.image_root_path, '_'.join(self.image_keys[idx].split(':')[i_b].split('_')[:-1]))).convert('RGB') 
          else:
              image = Image.open('{}/{}.jpg'.format(self.image_root_path, '_'.join(self.image_keys[idx].split('_')[:-1]))).convert('RGB')
          if len(np.array(image).shape) == 2:
            print('Wrong shape')
            image = np.tile(np.array(image)[:, :, np.newaxis], (1, 1, 3))
            image = Image.fromarray(image)
        
          region = image.crop(box=(x, y, x + w, y + h))
          region = self.transform(region)
          regions.append(region)
          if len(regions) == self.max_nregions:
              break
        region_mask[:len(regions)+1] = 1.  
    
        if len(regions) < self.max_nregions:
            if len(regions) == 0:
                print('Warning: empty image')
                regions = [torch.zeros((3, self.height, self.width)) for _ in range(self.max_nregions)]
            for _ in range(self.max_nregions-len(regions)):
                regions.append(torch.zeros(regions[0].size()))

        regions = torch.stack(regions, axis=0)
    
    region_mask = torch.FloatTensor(region_mask)
    return regions, region_mask

  def __len__(self):
    return len(self.keep_indices)

  def convert_to_fixed_length(self, mfcc):
    T = mfcc.shape[1] 
    pad = abs(self.max_nframes - T)
    if T < self.max_nframes:
      mfcc = np.pad(mfcc, ((0, 0), (0, pad)), 'constant', constant_values=(0))
    elif T > self.max_nframes:
      mfcc = mfcc[:, :-pad]
    return mfcc  
