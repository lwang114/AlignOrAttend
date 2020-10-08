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
from Image_phone_retrieval.steps.util import *

csv.field_size_limit(sys.maxsize)

def plot_attention(dataset, 
                   loader, 
                   audio_model, 
                   image_model, 
                   img_ids = None,
                   out_dir='./'): # TODO
  """ Generate the attention matrix learned by the word discovery model """
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  
  B = -1
  for i_b, batch in enumerate(loader):
    audio_input, image_input, nphones, nregions = batch
    D = image_input.size(-1)
    if i_b == 0:
      B = image_input.size(0)
    
    # Compute the dot-product attention weights
    for b in range(B):
      example_id = i_b * B + b
      img_id = dataset.image_keys[example_id]
      if img_ids is not None and not img_id in img_ids:
        continue
      else:
        print('Example {}'.format(example_id))
        audio_output = audio_model(audio_input[b].unsqueeze(0))
        image_output = image_model(image_input[b].unsqueeze(0))
        matchmap = torch.mm(image_output[b], audio_output[b]).t()
        attention = (matchmap[:nphones[b]]).softmax(-1).cpu().detach().numpy()  
        
        # Plot the attention weights
        fig, ax = plt.subplots(figsize=(30, 40))
        ax.invert_yaxis()

        plt.pcolor(attention, cmap=plt.cm.Greys, vmin=attention.min(), vmax=attention.max())
        cbar = plt.colorbar()
        for tick in cbar.ax.get_yticklabels():
          tick.set_fontsize(30)

        for tick in ax.get_xticklabels():
          tick.set_fontsize(30)
        # TODO Display word-level boundary labels
        plt.savefig('{}/attention_{}_{}.png'.format(out_dir, img_id, example_id))
  
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
     

def plot_clusters(embedding_vec_file, word_segment_file, n_clusters=10): # TODO
  """ Visualize the clusters discovered by the system using t-SNE """
  pass

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
  if not os.path.isdir(out_dir):
    os.mkdir(out_dir)
  audio_model_file = '{}/models/best_audio_model.pth'.format(args.exp_dir)
  image_model_file = '{}/models/best_image_model.pth'.format(args.exp_dir)

  audio_root_path_test = os.path.join(args.data_dir, 'val2014/wav/') 
  image_root_path_test = os.path.join(args.data_dir, 'val2014/imgs/')
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
  example_ids = indices[:10] # np.random.permutation(indices)[:10]
  img_ids = [dataset.image_keys[ex] for ex in example_ids]  
  '''
  plot_attention(dataset, 
                 loader, 
                 audio_model, 
                 image_model, 
                 out_dir=out_dir,
                 img_ids=img_ids)
  '''
  plot_boxes(dataset,
             loader,
             box_file='/ws/ifp-53_2/hasegawa/lwang114/data/mscoco/val2014/mscoco_val_bboxes_rcnn.json',
             data_dir='/ws/ifp-53_2/hasegawa/lwang114/data/mscoco/val2014/imgs/val2014',
             img_ids=img_ids)
