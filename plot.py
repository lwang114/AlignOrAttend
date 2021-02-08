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
from copy import deepcopy

plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.rc('axes', labelsize=20)
plt.rc('figure', titlesize=30)
plt.rc('font', size=30)

csv.field_size_limit(sys.maxsize)
NULL = '<NULL>'
def plot_attention(dataset, 
                   loader,
                   caption_file,
                   audio_model, 
                   image_model, 
                   img_ids = None,
                   out_dir='./',
                   att_type='norm_over_image'):
  """ Generate the attention matrix learned by the word discovery model """
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  
  B = -1
  
  captions = []
  with open(caption_file, 'r') as f:
    for line in f:
      captions.append(line.strip().split())
  
  ex = 0
  for i_b, batch in enumerate(loader):
    audio_input, image_input, nphones, nregions = batch
    D = image_input.size(-1)
    if i_b == 0:
      B = image_input.size(0)
    
    # Compute the dot-product attention weights
    for b in range(image_input.size(0)):
      example_id = dataset.keep_indices[i_b * B + b]
      img_id = dataset.image_keys[example_id]
      segmentation = [0] 
      segmentation.extend([seg[1] for seg in dataset.segmentations[example_id]])

      if img_ids is not None and not img_id in img_ids:
        continue
      else:
        ex += 1
        if len(img_ids) == ex:
          break
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
        fig, ax = plt.subplots(figsize=(15, 8))

        plt.pcolor(attention, cmap=plt.cm.Greys, vmin=attention.min(), vmax=attention.max())
        cbar = plt.colorbar()
        for tick in cbar.ax.get_yticklabels():
          tick.set_fontsize(30)

        ax.set_xticks(np.arange(attention.shape[0]), minor=False)
        ax.set_xticklabels(word_labels)
        ax.set_yticks(np.arange(attention.shape[1]), minor=False)
        
        ax.invert_yaxis()
        for tick in ax.get_xticklabels():
          tick.set_fontsize(30)

        for tick in ax.get_yticklabels():
          tick.set_fontsize(30)
          
        # Display word-level boundary labels
        plt.savefig('{}/attention_{}_{}_{}.png'.format(out_dir, att_type, img_id, example_id))
  
  
def plot_boxes(dataset,
               box_file,
               data_dir, 
               out_dir,
               caption_file=None,
               alignment_file=None,
               include_null=False,
               example_ids=None,
               ds_factor=1):
  """ Generate the boxes discovered by the RCNN """
  color = ['red', 'orange', 'yellow', 'green', 'blue-green', 'blue', 'purple', 'black']
  if example_ids is None:
    img_ids = dataset.image_keys
  else:
    img_ids = [dataset.image_keys[ex] for ex in example_ids]

  if caption_file and alignment_file:
    with open(caption_file, 'r') as capt_f,\
         open(alignment_file, 'r') as align_f:
        alignments = json.load(align_f)
        captions = [line.strip().split() for line in capt_f.read().split('\n')]
        print('Number of alignments, number of captions: {} {}'.format(len(alignments), len(captions)))

  if box_file.split('.')[-1] == 'json':
    with open(box_file, 'r') as fb:
      box_dict = json.load(fb)
  else:
    raise ValueError('Predicted bbox file is assumed to be of json format for now')
  
  for ex, img_id in zip(example_ids, img_ids):
    print(ex, img_id, img_id in box_dict)
    boxes = box_dict[img_id][:10]
    img_id = '_'.join(img_id.split('_')[:-1])
    im = cv2.imread('{}/{}.jpg'.format(data_dir, img_id))
    for i_box, box in enumerate(boxes[:10]):
        if caption_file and alignment_file:
            caption = captions[ex]
            alignment = alignments[ex // ds_factor]['alignment'] # XXX
            if include_null:
                caption = [NULL]+caption
            w_idx = alignment[i_box] 
            draw_bounding_box(im, box, labels=[caption[w_idx]], color=color[i_box % len(color)], font_scale=1.)
        else:
            draw_bounding_box(im, box, labels=[str(i_box)], color=color[i_box % len(color)], font_scale=1.)
        cv2.imwrite('{}/{}_annotated.jpg'.format(out_dir, img_id), im)

def filter_and_plot_boxes(dataset,
                          loader,
                          pred_box_file,
                          gold_box_file,
                          data_dir,
                          out_dir,
                          caption_file,
                          alignment_file,
                          include_null=False,
                          example_ids=None,
                          ds_factor=1):
  def _match_boxes(pred_boxes, gold_boxes):
    box_alignment = []
    # gold_labels = [' '.join(gbox[4:]) for gbox in gold_boxes]
    for pbox in pred_boxes:
      ious = []
      for gbox in gold_boxes:
        ious.append(_IoU(pbox[:4], gbox[:4]))

      if max(ious) == 0:
        box_alignment.append(-1)
      else:
        i_best = np.argmax(ious)
        box_alignment.append(i_best)
    return box_alignment

  def _IoU(pred, gold):
    if len(gold) == 2:
      p_start, p_end = pred[0], pred[1]
      g_start, g_end = gold[0], gold[1]
      i_start, u_start = max(p_start, g_start), min(p_start, g_start)  
      i_end, u_end = min(p_end, g_end), max(p_end, g_end)

      if i_start >= i_end:
        return 0.

      if u_start == u_end:
        return 1.

      iou = (i_end - i_start) / (u_end - u_start)
      assert iou <= 1 and iou >= 0
      return iou
    elif len(gold) == 4:
      xp_min, yp_min, xp_max, yp_max = pred
      xg_min, yg_min, xg_max, yg_max = gold
      xi_min, yi_min, xi_max, yi_max = max(xp_min, xg_min), max(yp_min, yg_min), min(xp_max, xg_max), min(yp_max, yg_max)
      xu_min, yu_min, xu_max, yu_max = min(xp_min, xg_min), min(yp_min, yg_min), max(xp_max, xg_max), max(yp_max, yg_max)
      
      if xi_min >= xi_max or yi_min >= yi_max:
        return 0.
      
      if xu_min == xu_max or yu_min == yu_max:
        return 1.

      Si = (xi_max - xi_min) * (yi_max - yi_min)
      Su = (xu_max - xu_min) * (yu_max - yu_min)
      iou = Si / Su
      assert iou <= 1 and iou >= 0
      return iou

  color = [(0, 0, 255), 
           (0, 0, 0), 
           (0, 130, 255), 
           (0, 255, 130),
           (0, 255, 0),
           (130, 255, 0),
           (255, 130, 0),
           (255, 0, 0),
           (255, 0, 130),
           (130, 0, 255)] # ['red', 'orange', 'yellow', 'green', 'blue-green', 'blue', 'purple', 'black']
          
  if example_ids is None:
    img_ids = dataset.image_keys
  else:
    img_ids = [dataset.image_keys[ex] for ex in example_ids]

  with open(caption_file, 'r') as capt_f,\
       open(alignment_file, 'r') as align_f:
    alignments = json.load(align_f)
    captions = [line.strip().split() for line in capt_f.read().split('\n')]
    print('Number of alignments, number of captions: {} {}'.format(len(alignments), len(captions)))

  # Load predicted boxes
  if pred_box_file.split('.')[-1] == 'json':
    with open(pred_box_file, 'r') as fb:
      pred_box_dict = json.load(fb)
  else:
    raise NotImplementedError('Predicted bbox file is assumed to be of json format for now')
  
  # Load gold boxes
  gold_box_dict = {}
  prev_img_id = ''
  with open(gold_box_file, 'r') as fg:
    for line_gold_box in fg:
      if len(gold_box_dict) == len(img_ids):
        break
      raw_gold_box_info = line_gold_box.strip().split()
      img_id = raw_gold_box_info[0]
      found = 0
      for cur_img_id in img_ids: 
        if '_'.join(cur_img_id.split('_')[:-1]) == img_id:
          found = 1
          break
  
      if found:
        x, y, w, h = raw_gold_box_info[-4:]
        x, y, w, h = int(x), int(y), int(w), int(h)
        # XXX Convert the box format from [x, y, w, h] to 
        # [x_min, y_min, x_max, y_max]
        box_info = [x, y, x+w, y+h, raw_gold_box_info[1]] 
        if prev_img_id != cur_img_id:
          gold_box_dict[cur_img_id] = [box_info]
          prev_img_id = cur_img_id
        else:
          gold_box_dict[cur_img_id].append(box_info)

  pred_f = open('{}/predicted_annotated_boxes.txt'.format(out_dir), 'w')
  for i_ex, (ex, img_id) in enumerate(zip(example_ids, img_ids)): 
    gold_boxes = gold_box_dict[img_id]
    pred_boxes = pred_box_dict[img_id][:10]
    caption = captions[ex]  
    if include_null:
      caption = [NULL]+caption
    alignment = alignments[i_ex // ds_factor]['alignment']
    # print(alignment, caption)

    # Append aligned captions to the predicted boxes
    for i_box, align_idx in enumerate(alignment):
      pred_boxes[i_box].append(caption[align_idx])

    # Filter boxes
    box_alignment = _match_boxes(pred_boxes, gold_boxes) 

    merged_boxes = {}    
    # Merge boxes
    for gold_box_idx, pred_box in zip(box_alignment, pred_boxes):
      if gold_box_idx == -1:
        continue

      if str(gold_box_idx) in merged_boxes:
        merged_boxes[str(gold_box_idx)][-1].append(pred_box[-1])
      else:
        merged_boxes[str(gold_box_idx)] = deepcopy(gold_boxes[gold_box_idx][:4])
        merged_boxes[str(gold_box_idx)].append([pred_box[-1]])

    img_id = '_'.join(img_id.split('_')[:-1])
    im = cv2.imread('{}/{}.jpg'.format(data_dir, img_id))
    for i_box, box in merged_boxes.items(): 
      start_pt = (box[0], box[1])
      end_pt = (box[2], box[3])
      cur_color = color[int(i_box) % len(color)]
      thickness = 2
      draw_bounding_box(im, box[:4], labels=[' '.join(box[4])], color=color[int(i_box) % len(color)]) 
      print('{}/{}.jpg, {}'.format(data_dir, img_id, cur_color))
      # im = cv2.rectangle(im, start_pt, end_pt, cur_color, 2)
      # im = cv2.putText(im, ' '.join(box[4]), (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cur_color, 1)
      # cv2.imwrite('{}/{}_filtered_annotated.jpg'.format(out_dir, img_id), im) 
      # pred_f.write('{} {} {} {} {} {}'.format(img_id, box[0], box[1], box[2], box[3], '_'.join(box[4])))

  pred_f.close()
    

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
 
def create_discovery_vs_retrieval_curve(discovery_result_file, retrieval_result_file, out_file):
    f1_vs_epoch = []
    recall_vs_epoch = []
    with open(discovery_result_file, 'r') as disc_f,\
         open(retrieval_result_file, 'r') as ret_f:
        for line in disc_f:
            parts = line.strip().split(', ')
            epoch = int(parts[0].split('.')[0].split('_')[-1][5:])
            f1_score = float(parts[-1].split('=')[-1])
            f1_vs_epoch.append((epoch, f1_score))

        for line in ret_f:
            parts = line.strip().split()
            epoch = -1
            recall = 0
            for i_part, part in enumerate(parts):
                if part[0] == '[': 
                    epoch = int(part.split(']')[0].split('[')[-1])
                elif part == 'R10:':
                    recall += float(parts[i_part+1])
            if epoch == -1:
              continue
            recall /= 2.
            recall_vs_epoch.append((epoch, recall))
    f1_vs_epoch = sorted(f1_vs_epoch, key=lambda x:x[0])
    recall_vs_epoch = sorted(recall_vs_epoch, key=lambda x:x[0])

    f1_vs_recall = {'Recall@10':[], 'Alignment F1':[], 'Epoch':[]}     
    for rec_ep in recall_vs_epoch:
        for f1_ep in f1_vs_epoch:
            if rec_ep[0] == f1_ep[0]:
                f1_vs_recall['Recall@10'].append(rec_ep[1])
                f1_vs_recall['Alignment F1'].append(f1_ep[1])
                f1_vs_recall['Epoch'].append(rec_ep[0])
    f1_vs_recall_df = pd.DataFrame(f1_vs_recall)
    f1_vs_recall_df.to_csv(out_file)

def create_smt_discovery_vs_retrieval_curve(discovery_result_file, retrieval_result_file, out_file, date=['2020-10-18', '2020-10-19']):
    f1_vs_epoch = []
    recall_vs_epoch = []
    with open(discovery_result_file, 'r') as disc_f,\
         open(retrieval_result_file, 'r') as ret_f:
        for line in disc_f:
            parts = line.strip().split(', ')
            epoch = int(parts[0].split('.')[0].split('_')[-1])
            f1_score = float(parts[-1].split('=')[-1])
            f1_vs_epoch.append((epoch, f1_score))

        epoch = -1
        for line in ret_f:
            parts = line.strip().split()
            cur_date = line.split()[0]
            if cur_date in date:
              if 'Iteration' in parts[2]:
                epoch = int(parts[3].split(',')[0])
                continue
              elif 'Image' in parts:
                recall_vs_epoch.append((epoch, float(parts[-1]))) 

    f1_vs_epoch = sorted(f1_vs_epoch, key=lambda x:x[0])
    recall_vs_epoch = sorted(recall_vs_epoch, key=lambda x:x[0])
    print(f1_vs_epoch, recall_vs_epoch)

    f1_vs_recall = {'Recall@10':[], 'Alignment F1':[], 'Epoch':[]}     
    for rec_ep in recall_vs_epoch:
        for f1_ep in f1_vs_epoch:
            if rec_ep[0] == f1_ep[0]:
                f1_vs_recall['Recall@10'].append(rec_ep[1])
                f1_vs_recall['Alignment F1'].append(f1_ep[1])
                f1_vs_recall['Epoch'].append(rec_ep[0])
    f1_vs_recall_df = pd.DataFrame(f1_vs_recall)
    f1_vs_recall_df.to_csv(out_file)

if __name__ == '__main__':
  import Image_phone_retrieval.models as models
  from Image_phone_retrieval.dataloaders.image_audio_caption_dataset_online import OnlineImageAudioCaptionDataset
  import argparse
  import os

  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--exp_dir', '-e', type=str, default='/ws/ifp-53_1/hasegawa/tools/espnet/egs/discophone/ifp_lwang114/magnet/Image_phone_retrieval/exp/biattentive_mml_rcnn_10_4_2020/tensor/mml')
  parser.add_argument('--data_dir', '-d', type=str, default='/ws/ifp-53_2/hasegawa/lwang114/data/mscoco/')
  parser.add_argument('--include_null', action='store_true')
  parser.add_argument('--task', '-t', type=int, default=0)
  parser.add_argument('--level', choices={'phone', 'word'}, default='word')
  parser.add_argument('--ds_factor', type=int, default=1)
  parser.add_argument('--nmt', action='store_true')
  args = parser.parse_args()
  
  out_dir = '{}/outputs/'.format(args.exp_dir)
  data_dir = '/ws/ifp-53_2/hasegawa/lwang114/data/mscoco'
  if not os.path.isdir(out_dir):
    os.mkdir(out_dir)
  audio_model_file = '{}/models/best_audio_model.pth'.format(args.exp_dir)
  image_model_file = '{}/models/best_image_model.pth'.format(args.exp_dir)

  audio_root_path_train = os.path.join(args.data_dir, 'train2014/wav/') 
  image_root_path_train = os.path.join(args.data_dir, 'train2014/imgs/')
  caption_file_train = os.path.join(args.data_dir, 'train2014/mscoco_train_text_captions.txt'\
                                    if args.level=='word' else 'train2014/mscoco_train_phone_captions_segmented.txt')
  segment_file_train = os.path.join(args.data_dir, 'train2014/mscoco_train_word_segments.txt')
  bbox_file_train = os.path.join(args.data_dir, 'train2014/mscoco_train_rcnn_feature.npz')
  
  audio_root_path_test = os.path.join(args.data_dir, 'val2014/wav/') 
  image_root_path_test = os.path.join(args.data_dir, 'val2014/imgs/')
  caption_file_test = os.path.join(args.data_dir, 'val2014/mscoco_val_{}_captions.txt'.format(args.level))
  segment_file_test = os.path.join(args.data_dir, 'val2014/mscoco_val_word_segments.txt')
  bbox_file_test = os.path.join(args.data_dir, 'val2014/mscoco_val_rcnn_feature.npz')
  split_file = os.path.join(args.data_dir, 'val2014/mscoco_val_split.txt')

  if args.task == 0:
    dataset = OnlineImageAudioCaptionDataset(audio_root_path_test,
                                          image_root_path_test,
                                          segment_file_test,
                                          bbox_file_test,
                                          keep_index_file=split_file,
                                          configs={})
  else:
    dataset = OnlineImageAudioCaptionDataset(audio_root_path_train,
                                          image_root_path_train,
                                          segment_file_train,
                                          bbox_file_train,
                                          keep_index_file=None,
                                          configs={})
  loader = torch.utils.data.DataLoader(dataset, 
                                       batch_size=15, 
                                       shuffle=False, 
                                       num_workers=0, 
                                       pin_memory=True)
  audio_model = models.Davenet(embedding_dim=1024)
  image_model = models.LinearTrans(input_dim=2048, embedding_dim=1024)

  indices = np.arange(1000)
  example_ids = [k for k, img_key in enumerate(dataset.image_keys) if '_'.join(img_key.split('_')[:-1]) == 'COCO_train2014_000000262399']
  # dataset.keep_indices[ex] for ex in indices] # XXX # np.random.permutation(indices)[:10]
  img_ids = [dataset.image_keys[ex] for ex in example_ids] 
  # print(example_ids, img_ids)

  if args.task == 0:
      filter_and_plot_boxes(dataset,
                 loader,
                 pred_box_file='{}/val2014/mscoco_val_bboxes_rcnn.json'.format(data_dir),
                 gold_box_file='{}/val2014/mscoco_val_bboxes.txt'.format(data_dir),
                 data_dir='{}/val2014/imgs/val2014'.format(data_dir),
                 out_dir=out_dir, 
                 alignment_file='{}/alignment.json'.format(args.exp_dir),
                 caption_file=caption_file_test,
                 example_ids=example_ids,
                 include_null=args.include_null,
                 ds_factor=1)
  elif args.task == 1:
    # SMT phone path: /ws/ifp-53_2/hasegawa/lwang114/fall2020/exp/cont_mixture_mscoco_phone_rcnn_10_1_2020/
    # DAVEnet audio path: /ws/ifp-53_1/hasegawa/tools/espnet/egs/discophone/ifp_lwang114/magnet/Image_phone_retrieval/exp/biattentive_mml_rcnn_10_4_2020/tensor/mml
    # SMT path: /ws/ifp-53_2/hasegawa/lwang114/fall2020/exp/cont_mixture_mscoco_davenet_rcnn_10region_topwordonly_10_12_2020/
    plot_boxes(dataset,
             box_file='{}/train2014/mscoco_train_bboxes_rcnn.json'.format(data_dir),
             data_dir='{}/train2014/imgs/train2014'.format(data_dir),
             out_dir=out_dir,
             alignment_file='{}/alignment.json'.format(args.exp_dir),
             caption_file=caption_file_train,
             example_ids=example_ids,
             include_null=args.include_null,
             ds_factor=args.ds_factor)
  elif args.task == 2:
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
  elif args.task == 3:
      plot_clusters(embedding_vec_file='/ws/ifp-53_1/hasegawa/tools/espnet/egs/discophone/ifp_lwang114/magnet/Image_phone_retrieval/exp/mscoco_val_davenet_features/test_features_300dim.npz', # TODO
                word_segment_file='{}/val2014/mscoco_val_word_segments.txt'.format(data_dir),
                class2idx_file='{}/val2014/class2idx.json'.format(data_dir),
                ds_ratio=16,
                out_file='{}/word_clusters'.format(args.exp_dir)) # TODO
  elif args.task == 4:
    if args.nmt:
      retrieval_result_file = '{}/result_file.txt'.format(args.exp_dir)
      discovery_result_file = '{}/alignment_results.txt'.format(args.exp_dir)
    else:
      retrieval_result_file = '{}/train.log'.format(args.exp_dir)
      discovery_result_file = '{}/alignment_results.txt'.format(args.exp_dir)

    out_file = '{}/discovery_vs_retrieval.csv'.format(args.exp_dir)
    
    create_smt_discovery_vs_retrieval_curve(discovery_result_file, retrieval_result_file, out_file)
