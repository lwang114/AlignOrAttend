import torch
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import numpy as np
import csv
from PIL import Image
import cv2
from opencv_draw_annotation import draw_bounding_box
import sys
import base64
import zlib
import time
import mmap
import sklearn
from sklearn.manifold import TSNE
import seaborn as sns; sns.set()
import pandas as pd
from Image_phone_retrieval.steps.util import *

csv.field_size_limit(sys.maxsize)

def plot_attention(dataset, 
                   loader,
                   caption_file,
                   audio_model, 
                   image_model, 
                   img_ids = None,
                   out_dir='./',
                   att_type='norm_over_image'): # TODO
  """ Generate the attention matrix learned by the word discovery model """
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  
  B = -1
  
  captions = []
  with open(caption_file, 'r') as f:
    for line in f:
      captions.append(line.strip().split())
  
  for i_b, batch in enumerate(loader):
    audio_input, image_input, nphones, nregions = batch
    D = image_input.size(-1)
    if i_b == 0:
      B = image_input.size(0)
    
    # Compute the dot-product attention weights
    for b in range(B):
      example_id = dataset.keep_indices[i_b * B + b]
      img_id = dataset.image_keys[example_id]
      segmentation = [0] 
      segmentation.extend([seg[1] for seg in dataset.segmentations[example_id]])

              
      if img_ids is not None and not img_id in img_ids:
        continue
      else:
        print('Example {}'.format(example_id))
        audio_output = audio_model(audio_input[b].unsqueeze(0))
        pooling_ratio = round(audio_input.size(-1) / audio_output.size(-1))

        caption = captions[example_id]
        word_labels = ['']*audio_output.size(-1)
        dur = round(float(segmentation[-1]) / (10 * pooling_ratio))
        start = -1
        for w, start_ms, end_ms in zip(caption, segmentation[:-1], segmentation[1:]):
          start_ms, end_ms = float(start_ms), float(end_ms)
          cur_start = int(start_ms / (10 * pooling_ratio))
          if cur_start == start:
            cur_start = start + 1
          start = cur_start
          word_labels[start] = w
        
        image_output = image_model(image_input[b].unsqueeze(0))
        matchmap = torch.mm(image_output.squeeze(0), audio_output.squeeze(0)).t()
        if att_type == 'norm_over_image':
          attention = (matchmap[:dur]).softmax(-1).cpu().detach().numpy()  
        elif att_type == 'norm_over_audio':
          attention = (matchmap[:dur]).softmax(0).cpu().detach().numpy()
          
        # Plot the attention weights
        fig, ax = plt.subplots(figsize=(8, 15))

        plt.pcolor(attention, cmap=plt.cm.Greys, vmin=attention.min(), vmax=attention.max())
        cbar = plt.colorbar()
        for tick in cbar.ax.get_yticklabels():
          tick.set_fontsize(30)

        ax.set_xticks(np.arange(attention.shape[1]), minor=False)
        ax.set_yticks(np.arange(attention.shape[0]), minor=False)
        ax.set_yticklabels(word_labels)
        
        ax.invert_yaxis()
        for tick in ax.get_xticklabels():
          tick.set_fontsize(30)

        for tick in ax.get_yticklabels():
          tick.set_fontsize(30)
          
        # Display word-level boundary labels
        plt.savefig('{}/attention_{}_{}_{}.png'.format(out_dir, att_type, img_id, example_id))
  
  return image_ids
  
def plot_boxes(dataset,
               loader, 
               box_file,
               data_dir, 
               img_ids=None):
  """ Generate the boxes discovered by the RCNN """
  color = ['red', 'orange', 'yellow', 'green', 'blue-green', 'blue', 'purple', 'black']
  if img_ids is None:
    img_ids = dataset.image_keys
  
  '''  
  if box_file.split('.')[-1] == 'tsv':
    FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']
    with open(box_file, 'r+b') as tsv_in_file:
      reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
      
      for item in reader:
        img_id = int(item['image_id'])
        found = 0
        for cur_img_id in img_ids:
          if int(cur_img_id.split('_')[2]) == img_id:
            img_id = cur_img_id
            found = 1

        if found:
          print(img_id)
          n_boxes = int(item['num_boxes'])
          boxes = np.frombuffer(base64.decodestring(item['boxes']),
                                      dtype=np.float32).reshape((n_boxes, -1))
          # Load the image according to the image ids
          im = Image.open('{}/{}.jpg'.format(data_dir, img_id))
          for i_box, box in enumerate(boxes[:10]):
            draw_bounding_box(im, box, labels=[str(i_box)], color=color[i_box % len(color)])
            cv2.imwrite('{}/{}_annotated.jpg'.format(out_dir, img_id), im)
  '''
  if box_file.split('.')[-1] == 'json':
    with open(box_file, 'r') as fb:
      box_dict = json.load(fb)
  
  for img_id in img_ids:
    print(img_id, img_id in box_dict)
    boxes = box_dict[img_id][:10]
    img_id = '_'.join(img_id.split('_')[:-1])
    im = cv2.imread('{}/{}.jpg'.format(data_dir, img_id))
    for i_box, box in enumerate(boxes[:10]):
        draw_bounding_box(im, box, labels=[str(i_box)], color=color[i_box % len(color)])
        cv2.imwrite('{}/{}_annotated.jpg'.format(out_dir, img_id), im)
     
def plot_clusters(embedding_vec_file, 
                  word_segment_file, 
                  class2idx_file, 
                  ds_ratio, 
                  out_file,
                  word_freq_file=None,
                  max_cluster_size=500,
                  n_clusters=10):
  """ Visualize the clusters discovered by the system using t-SNE """
  PERSON_S_CLASS = ['man', 'woman', 'boy', 'girl', 'child']
  P2S = {'men':'man', 'women':'woman', 'boys':'boy', 'girls':'girl', 'children':'child'}
  colors = 'rgbkmy'
  dot = 'xo-' 

  with open(class2idx_file, 'r') as f:
    class2idx = json.load(f)
    for c_s in PERSON_S_CLASS:
      self.class2idx[c_s] = len(self.class2idx)
 
  if not word_freq_file:
    freqs = {c:0 for c in class2idx}
    with open('{}_freqs.json'.format(out_file), 'w') as freq_f,\
         open(word_segment_file, 'r') as seg_f:
      for line in seg_f:
        if w in class2idx:
          freqs[class2idx[w]] += 1
        elif w in P2S:
          freqs[class2idx[P2S[w]]] += 1
      json.dump(freqs, freq_f, indent=4, sort_keys=True)
  else:
    with open('{}_freqs.json'.format(out_file), 'r') as freq_f:
      freqs = json.load(f)
  
  classes = sorted(class2idx, key=lambda x:freqs[x], reverse=True)[:n_clusters]     
  cluster_sizes = {c:0 for c in classes}
  embedding_vec_npz = np.load(embedding_vec_file)
  ex = 0
  cur_capt_id = ''
  embeddings = []
  labels = []
  sent = []
  with open(word_segment_file, 'r') as seg_f:
      for line in seg_f:
        parts = line.split()
        capt_id, w, start, end = parts

        if capt_id != cur_capt_id:
          cur_capt_id = capt_id
     
        parts = line.split()
        capt_id, w, start, end = parts

        if capt_id != cur_capt_id:
          cur_capt_id = capt_id
          sent = [w]
          ex += 1
        else:
          sent.append(w)
        
        if not w in class2idx and not w in P2S:
          continue
        elif w in P2S:
          w = P2S[w] 
        
        if cluster_sizes[w] >= max_cluster_size:
          continue
        else:
          cluster_sizes[w] += 1  
          embeddings.append(embedding_vec_npz['arr_{}'.format(ex)][len(sent)-1])
          labels.append(w)

  X = np.concatenate(embeddings, axis=0)
  tsne = TSNE(n_components=len(cluster2idx))
  X_embedded = tsne.fit_transform(X)
  cluster_dfs = pd.DataFrame({'x': X_embedded[:, 0],
                              'y': X_embedded[:, 1],
                              'cluster_label': y})
  cluster_dfs.to_csv('{}_dataframe.csv'.format(out_file), cluster_dfs)
  sns.scatterplot(data=cluster_dfs, x='x', y='y', hue='cluster_label', stype='cluster_label') 
  plt.savefig('{}_tsne_plot.png'.format(out_file))
 

if __name__ == '__main__':
  import Image_phone_retrieval.models as models
  from Image_phone_retrieval.dataloaders.image_audio_caption_dataset_online import OnlineImageAudioCaptionDataset
  import argparse
  import os

  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--exp_dir', '-e', type=str, default='/ws/ifp-53_1/hasegawa/tools/espnet/egs/discophone/ifp_lwang114/magnet/Image_phone_retrieval/exp/biattentive_mml_rcnn_10_4_2020/tensor/mml')
  parser.add_argument('--data_dir', '-d', type=str, default='/ws/ifp-53_2/hasegawa/lwang114/data/mscoco/')
  args = parser.parse_args()
  
  out_dir = '{}/outputs/'.format(args.exp_dir)
  data_dir = '/ws/ifp-53_2/hasegawa/lwang114/data/mscoco'
  if not os.path.isdir(out_dir):
    os.mkdir(out_dir)
  audio_model_file = '{}/models/best_audio_model.pth'.format(args.exp_dir)
  image_model_file = '{}/models/best_image_model.pth'.format(args.exp_dir)

  audio_root_path_test = os.path.join(args.data_dir, 'val2014/wav/') 
  image_root_path_test = os.path.join(args.data_dir, 'val2014/imgs/')
  caption_file_test = os.path.join(args.data_dir, 'val2014/mscoco_val_text_captions.txt')
  segment_file_test = os.path.join(args.data_dir, 'val2014/mscoco_val_word_segments.txt')
  bbox_file_test = os.path.join(args.data_dir, 'val2014/mscoco_val_rcnn_feature.npz')
  split_file = os.path.join(args.data_dir, 'val2014/mscoco_val_split.txt')
  
  dataset = OnlineImageAudioCaptionDataset(audio_root_path_test,
                                          image_root_path_test,
                                          segment_file_test,
                                          bbox_file_test,
                                          keep_index_file=split_file,
                                          configs={})
  loader = torch.utils.data.DataLoader(dataset, 
                                       batch_size=15, 
                                       shuffle=False, 
                                       num_workers=0, 
                                       pin_memory=True)
  audio_model = models.Davenet(embedding_dim=1024)
  image_model = models.LinearTrans(input_dim=2048, embedding_dim=1024)

  indices = np.arange(1000)
  example_ids = [dataset.keep_indices[ex] for ex in indices[:10]] # np.random.permutation(indices)[:10]
  img_ids = [dataset.image_keys[ex] for ex in example_ids]  
  plot_attention(dataset, 
                 loader, 
                 caption_file_test,
                 audio_model, 
                 image_model, 
                 out_dir=out_dir,
                 img_ids=img_ids)
  plot_attention(dataset, 
                 loader, 
                 caption_file_test,
                 audio_model, 
                 image_model, 
                 out_dir=out_dir,
                 img_ids=img_ids,
                 att_type='norm_over_audio')
  plot_boxes(dataset,
             loader,
             box_file='{}/val2014/mscoco_val_bboxes_rcnn.json'.format(data_dir),
             data_dir='{}/val2014/imgs/val2014'.format(data_dir),
             img_ids=img_ids)
  '''
  plot_clusters(embedding_vec_file='/ws/ifp-53_1/hasegawa/tools/espnet/egs/discophone/ifp_lwang114/magnet/Image_phone_retrieval/exp/mscoco_val_davenet_features/test_features_300dim.npz', # TODO
                word_segment_file='{}/val2014/mscoco_val_word_segments.txt'.format(data_dir),
                class2idx_file='{}/val2014/class2idx.json'.format(data_dir),
                ds_ratio=16,
                out_file='{}/word_clusters'.format(args.exp_dir)) # TODO
  '''
