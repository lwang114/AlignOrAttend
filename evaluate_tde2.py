import numpy as np
import json
import os
import argparse
import logging
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import time
from copy import deepcopy
import functools

PUNCT = [',', '\'', '\"', '/', '?', '>', '<', '#', '%', '&', '*', ':', ';', '!', '.']
STOP = stopwords.words('english') + ['er', 'oh', 'ah', 'uh', 'um', 'ha']
PERSON_S_CLASS = ['man', 'woman', 'boy', 'girl', 'child', 'player']
P2S = {'people':'person', 
       'couple':'person', 
       'men':'man', 
       'women':'woman', 
       'boys':'boy', 
       'girls':'girl', 
       'children':'child',
       'group':'person', 
       'players':'player',
       'bikes': 'bike',
       'cars': 'car',
       'glasses': 'glass'}
W2C = {'television': 'tv', 
       'plant': 'potted_plant',
       'glove': 'baseball_glove',
       'bat': 'baseball_bat',
       'table': 'dining_table',
       'hydrant': 'fire_hydrant',
       'drier': 'hair_drier',
       'hot': 'hot_dog',
       'dog': 'hot_dog',
       'meter': 'parking_meter',
       'sign': 'stop_sign',
       'ball': 'sports_ball',
       'light': 'traffic_light',
       'traffic': 'traffic_light',
       'glass': 'wine_glass',
       'bike': 'bicycle'}
C2W = {c:w for w, c in W2C.items()}
NULL = '<NULL>'
NULL_SEGMENT = '0_0'

def alignmentF1(pred_file,
                gold_file):
  # Alignment recall is the percentage of discovered alignments in the set of true alignments 
  # and alignment precision is the percentage of true alignments in the set of discovered alignments
  # ------
  # Inputs:
  # ------
  #   pred_file: text file of the following format: 
  #       [image_id]_1 [x_min] [y_min] [x_max] [y_max] [caption label 1]_..._[caption label K] (optional: [scores])
  #       ...
  #       [image_id]_N [x_min] [y_min] [x_max] [y_max] [caption label 1]_..._[caption label K] (optional: [scores])
  #
  #   gold_file: text file of the following format:
  #       [image_id]_1 [x_min] [y_min] [x_max] [y_max] [caption label]
  #       ...
  #       [image_id]_N [x_min] [y_min] [x_max] [y_max] [caption label]
  # -------
  # Outputs:
  # -------
  #   precision, recall, f1: floats in the range [0, 1]
  gold_alignments = []
  pred_alignments = []
  with open(pred_file, 'r') as pred_f,\
       open(gold_file, 'r') as gold_f:
    cur_id = ''
    for line in pred_f:
      parts = line.split()
      img_id = parts[0]
      x_min = int(parts[1])
      y_min = int(parts[2])
      x_max = int(parts[3])
      y_max = int(parts[4])
      pred_labels = parts[5].split('_')
      if img_id != cur_id:
        pred_alignments.append([(img_id, x_min, y_min, x_max, y_max, pred_labels)])
        cur_id = img_id
      else:
        pred_alignments[-1].append((img_id, x_min, y_min, x_max, y_max, pred_labels))
    
    cur_id = ''
    for line in gold_f:
      parts = line.split()
      img_id = parts[0]
      x_min = int(parts[1])
      y_min = int(parts[2])
      x_max = int(parts[3])
      y_max = int(parts[4])
      gold_label = parts[5]
      if img_id != cur_id:
        gold_alignments.append([(img_id, x_min, y_min, x_max, y_max, gold_label)])
        cur_id = img_id
      else:
        gold_alignments[-1].append((img_id, x_min, y_min, x_max, y_max, gold_label))

  correct = 0 
  n_pred = 0
  n_gold = 0
  for pred_alignments_i, gold_alignments_i in zip(pred_alignments, gold_alignments):
    n_gold += len([gold for gold in gold_alignments_i if gold[5] != NULL])

    for pred in pred_alignments_i:
      n_pred += 1
      for gold in gold_alignments_i:
        if functools.reduce(lambda x, y: x and y, map(lambda g, p: g==p, gold[1:5], pred[1:5]), True):
          gold_label = gold[5]
          pred_labels = pred[5]
          if gold_label in pred_labels:
            # print(gold, pred)
            correct += 1 
          else:
            pred_class_labels = [W2C[pred_label] for pred_label in pred_labels if pred_label in W2C]
            if gold_label in pred_class_labels:
              correct += 1
  
  print('{} corrects, {} gold alignments, {} pred alignments'.format(correct, n_gold, n_pred))
  recall = correct / n_gold
  precision = correct / n_pred
  f1 = 2 * recall * precision / (recall + precision) if recall + precision > 0  else 0
  return recall, precision, f1

def segmentAlignmentF1(pred_file,
                       gold_file):
  # Alignment recall is the percentage of discovered alignments in the set of true alignments 
  # and alignment precision is the percentage of true alignments in the set of discovered alignments.
  # The only difference between this function and the alignment is that a segment is considered 
  # ``discovered`` if a predicted segment with the same label lies in its span
  # ------
  # Inputs:
  # ------
  #   pred_file: text file of the following format: 
  #       [image_id]_1 [x_min] [y_min] [x_max] [y_max] [caption label 1]_..._[caption label K] (optional: [scores])
  #       ...
  #       [image_id]_N [x_min] [y_min] [x_max] [y_max] [caption label 1]_..._[caption label K] (optional: [scores])
  #
  #   gold_file: text file of the following format:
  #       [image_id]_1 [x_min] [y_min] [x_max] [y_max] [caption label]
  #       ...
  #       [image_id]_N [x_min] [y_min] [x_max] [y_max] [caption label]
  # -------
  # Outputs:
  # -------
  #   precision, recall, f1: floats in the range [0, 1]
  def _find_segment(p, g):
      p_times = p.split(',')
      g_times = g.split(',')
      p_start = float(p_times[0])
      p_end = float(p_times[1])
      g_start = float(g_times[0])
      g_end = float(g_times[1])
      return (p_start >= g_start) and (p_end <= g_end) 

  gold_alignments = []
  pred_alignments = []
  with open(pred_file, 'r') as pred_f,\
       open(gold_file, 'r') as gold_f:
    cur_id = ''
    for line in pred_f:
      parts = line.split()
      img_id = parts[0]
      x_min = int(parts[1])
      y_min = int(parts[2])
      x_max = int(parts[3])
      y_max = int(parts[4])
      pred_labels = parts[5].split('_')
      if img_id != cur_id:
        pred_alignments.append([(img_id, x_min, y_min, x_max, y_max, pred_labels)])
        cur_id = img_id
      else:
        pred_alignments[-1].append((img_id, x_min, y_min, x_max, y_max, pred_labels))
    
    cur_id = ''
    for line in gold_f:
      parts = line.split()
      img_id = parts[0]
      x_min = int(parts[1])
      y_min = int(parts[2])
      x_max = int(parts[3])
      y_max = int(parts[4])
      gold_label = parts[5]
      if img_id != cur_id:
        gold_alignments.append([(img_id, x_min, y_min, x_max, y_max, gold_label)])
        cur_id = img_id
      else:
        gold_alignments[-1].append((img_id, x_min, y_min, x_max, y_max, gold_label))

  correct = 0 
  n_pred = 0
  n_gold = 0
  for pred_alignments_i, gold_alignments_i in zip(pred_alignments, gold_alignments):
    n_gold += len(gold for gold in gold_alignments_i if gold[5] != NULL_SEGMENT)

    for pred in pred_alignments_i:
      n_pred += 1
      for gold in gold_alignments_i:
        if functools.reduce(lambda x, y: x and y, map(lambda g, p: g==p, gold[1:5], pred[1:5]), True):
          gold_label = gold[5]
          pred_labels = pred[5]
          if functools.reduce(lambda x, y: x or y, map(lambda p: _find_segment(p, gold_label), pred_labels), True):
            # print(gold, pred)
            correct += 1 
          else:
            pred_class_labels = [W2C[pred_label] for pred_label in pred_labels if pred_label in W2C]
            if functools.reduce(lambda x, y: x or y, map(lambda p: _find_segment(p, gold_label), pred_labels), True):
              correct += 1
  
  print('{} corrects, {} gold alignments, {} pred alignments'.format(correct, n_gold, n_pred))
  recall = correct / n_gold
  precision = correct / n_pred
  f1 = 2 * recall * precision / (recall + precision) if recall + precision > 0  else 0
  return recall, precision, f1



def create_gold_annotated_box_file(caption_file, 
                                   box_file,
                                   out_file='gold_annotated_box.txt',
                                   keep_id_file=None):
  # Create a annotated box file in the format:
  #       [image_id] [x_min] [y_min] [x_max] [y_max] [caption label]
  #       ...
  #       [image_id] [x_min] [y_min] [x_max] [y_max] [caption label]
  # ------
  # Inputs:
  # ------
  #   caption_file: text file, each line containing a caption with format:
  #                 [word_1] [word_2]...[word_n]
  #   box_file: text file, each line containing a box with format:
  #             [image_id] [label] [x] [y] [width] [height]
  #   class2idx_file: json file containing the dictionary 
  #                   {c:i_c for i_c, c in enumerate(classes)}
  gold_annotated_boxes = []
  with open(caption_file, 'r') as capt_f,\
       open(box_file, 'r') as box_f:
    captions = [line.strip().split() for line in capt_f]
    if keep_id_file: 
      with open(keep_id_file, 'r') as keep_f:
        keep_indices = [ex for ex, line in enumerate(keep_f) if int(line)]
      
    cur_id = ''
    ex = -1
    for line in box_f:
      parts = line.strip().split()
      img_id = parts[0]
      class_label = parts[1]
      x = int(parts[2])
      y = int(parts[3])
      w = int(parts[4])
      h = int(parts[5])
     
      if img_id != cur_id:
        cur_id = img_id
        ex += 1
        if not ex in keep_indices:
          continue
        # if ex >= 100: # XXX
        #   break
        label = NULL
        for word in captions[ex]:
          if class_label == word:
            label = word 
          elif class_label in C2W and word in C2W[class_label]:
            label = word
          elif class_label == 'person' and (word in PERSON_S_CLASS):
            label = word
          elif word in P2S:
            word_s = P2S[word]
            if word_s in PERSON_S_CLASS or word_s in C2W:
              label = word
        # print(ex, captions[ex], class_label, label) # XXX
        gold_annotated_boxes.append([(img_id, x, y, x+w, y+h, label)])
      
      else:
        if not ex in keep_indices:
          continue
        label = NULL
        for word in captions[ex]:
          if class_label == word:
            label = word 
          elif class_label in C2W and word in C2W[class_label]:
            label = word
          elif class_label == 'person' and (word in PERSON_S_CLASS):
            label = word
          elif word in P2S:
            word_s = P2S[word]
            if word_s in PERSON_S_CLASS or word_s in C2W:
              label = word
        # print(ex, captions[ex], class_label, label) # XXX
        gold_annotated_boxes[-1].append((img_id, x, y, x+w, y+h, label))
  
  with open(out_file, 'w') as out_f:
    for gold_boxes_i in gold_annotated_boxes:
      for gold_box in gold_boxes_i:
        out_f.write('{} {} {} {} {} {}\n'.format(gold_box[0],\
                                                 gold_box[1],\
                                                 gold_box[2],\
                                                 gold_box[3],\
                                                 gold_box[4],\
                                                 gold_box[5]))

def create_gold_annotated_segment_box_file(caption_file, 
                                           box_file,
                                           out_file='gold_annotated_box.txt',
                                           keep_id_file=None):
  # Create a annotated box file in the format:
  #       [image_id] [x_min] [y_min] [x_max] [y_max] [start frame],[end frame]
  #       ...
  #       [image_id] [x_min] [y_min] [x_max] [y_max] [start frame],[end frame]
  # ------
  # Inputs:
  # ------
  #   caption_file: text file, each line containing a segment with format:
  #                 [image_id] [label] [start frame] [end frame]
  #   box_file: text file, each line containing a box with format:
  #             [image_id] [label] [x] [y] [width] [height]
  #   class2idx_file: json file containing the dictionary 
  #                   {c:i_c for i_c, c in enumerate(classes)}
  gold_annotated_boxes = []
  with open(caption_file, 'r') as capt_f,\
       open(box_file, 'r') as box_f:
    if keep_id_file: 
      with open(keep_id_file, 'r') as keep_f:
        keep_indices = [ex for ex, line in enumerate(keep_f) if int(line)]

    cur_id = ''
    ex = -1
    captions = []
    for line_capt in capt_f:
      capt_id, word, start, end = line_capt.strip().split()
      start, end = int(start), int(end)
      if capt_id != cur_id:
        cur_id = capt_id
        ex += 1
        if not ex in keep_indices:
            continue
        captions.append([(capt_id, start, end, word)])
      else:
        if not ex in keep_indices:
            continue
        captions[-1].append((capt_id, start, end, word))

    cur_id = ''
    ex = -1
    for line in box_f:
      parts = line.strip().split()
      img_id = parts[0]
      class_label = parts[1]
      x = int(parts[2])
      y = int(parts[3])
      w = int(parts[4])
      h = int(parts[5])
     
      if img_id != cur_id:
        cur_id = img_id
        ex += 1
        if not ex in keep_indices:
          continue
        # if ex >= 100: # XXX
        #   break
        label = NULL_SEGMENT
        for segment in captions[ex]:
          start = segment[1]
          end = segment[2]
          word = segment[3]
          if class_label == word:
            label = '{},{}'.format(start, end) 
          elif class_label in C2W and word in C2W[class_label]:
            label = '{},{}'.format(start, end)
          elif class_label == 'person' and (word in PERSON_S_CLASS):
            label = '{},{}'.format(start, end)
          elif word in P2S:
            word_s = P2S[word]
            if word_s in PERSON_S_CLASS or word_s in C2W:
              label = '{},{}'.format(start, end)
        # print(ex, captions[ex], class_label, label) # XXX
        gold_annotated_boxes.append([(img_id, x, y, x+w, y+h, label)])
      else:
        if not ex in keep_indices:
          continue
        label = NULL_SEGMENT
        for segment in captions[ex]:
          start = segment[1]
          end = segment[2]
          word = segment[3]
          if class_label == word:
            label = '{},{}'.format(start, end) 
          elif class_label in C2W and word in C2W[class_label]:
            label = '{},{}'.format(start, end)
          elif class_label == 'person' and (word in PERSON_S_CLASS):
            label = '{},{}'.format(start, end)
          elif word in P2S:
            word_s = P2S[word]
            if word_s in PERSON_S_CLASS or word_s in C2W:
              label = '{},{}'.format(start, end)
        # print(ex, captions[ex], class_label, label) # XXX
        gold_annotated_boxes[-1].append((img_id, x, y, x+w, y+h, label))

  with open(out_file, 'w') as out_f:
    for gold_boxes_i in gold_annotated_boxes:
      for gold_box in gold_boxes_i:
        out_f.write('{} {} {} {} {} {}\n'.format(gold_box[0],\
                                                 gold_box[1],\
                                                 gold_box[2],\
                                                 gold_box[3],\
                                                 gold_box[4],\
                                                 gold_box[5]))
 
    
def filter_boxes(caption_file, 
                 pred_alignment_file, 
                 pred_box_file, 
                 gold_box_file,
                 out_file,
                 include_null=False,
                 keep_id_file=None):
  # Filter predicted boxes by merging boxes matched to the same true boxes
  # and create a predicted box file of the format:
  #   [image_id]_1 [x_min] [y_min] [x_max] [y_max] [caption_label 1]_..._[caption label K] (optional: [scores]) 
  # ------
  # Inputs:
  # ------
  #   caption_file: text file, each line containing a caption with format: 
  #                [word_1] [word_2]...[word_n] 
  #   pred_alignment_file: json file storing the alignment
  #   pred_box_file: text file, each line containing a box with format:
  #                 [image_id] [label] [x] [y] [width] [height] 
  #   gold_box_file:  text file, each line containing a box with format:
  #                 [image_id] [label] [x] [y] [width] [height] 
  #   keep_id_file (optional): text file where example i is used if the i-th line is '1' and not used if the i-th
  #                            line is '0'
  # -------
  # Outputs:
  # -------
  #   merged_boxes: a list of boxes,
  #     [[[x_min_b, y_min_b, x_max_b, y_max_b, labels_b] for b in range(n_boxes_i)] for i in range(n)]
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
         
  with open(caption_file, 'r') as capt_f,\
       open(pred_alignment_file, 'r') as align_f:
    alignments = json.load(align_f)

    if keep_id_file:
      with open(keep_id_file, 'r') as keep_f:
        keep_indices = [ex for ex, line in enumerate(keep_f) if int(line)]
    
    if keep_id_file:
      captions = [line.strip().split() for ex, line in enumerate(capt_f) if ex in keep_indices]
    else:
      captions = [line.strip().split() for line in capt_f]
  
  # Load gold boxes
  gold_box_dict = {}
  prev_img_id = ''
  img_ids = []
  ex = -1
  with open(gold_box_file, 'r') as fg:
    for line_gold_box in fg:
      raw_gold_box_info = line_gold_box.strip().split()
      cur_img_id = raw_gold_box_info[0]
      x, y, w, h = raw_gold_box_info[-4:]
      x, y, w, h = int(x), int(y), int(w), int(h)
      
      # Convert the box format from [x, y, w, h] to 
      # [x_min, y_min, x_max, y_max]
      box_info = [x, y, x+w, y+h, raw_gold_box_info[1]]  
      if prev_img_id != cur_img_id:
        prev_img_id = cur_img_id
        ex += 1
        if not ex in keep_indices:
          continue
        # if ex >= 100: # XXX
        #   break
        gold_box_dict[cur_img_id] = [box_info]
        img_ids.append(cur_img_id)
        prev_img_id = cur_img_id 
      else:
        if not ex in keep_indices:
          continue
        gold_box_dict[cur_img_id].append(box_info)
  print('Number of gold boxes: {}'.format(len(gold_box_dict)))

  # Load predicted boxes
  if pred_box_file.split('.')[-1] == 'json':
    with open(pred_box_file, 'r') as fb:
      pred_box_dict = json.load(fb)
      pred_box_ids = sorted(pred_box_dict, key=lambda x:int(x.split('_')[-1]))
      imgid2boxid = {img_id:box_id for img_id in img_ids for box_id in pred_box_ids if '_'.join(box_id.split('_')[:-1]) == img_id}
  else:
    raise NotImplementedError('Predicted bbox file is assumed to be of json format for now')

  pred_f = open(out_file, 'w')
  merged_boxes_all = []
  for img_id, caption, align_info in zip(img_ids, captions, alignments):
    gold_boxes = gold_box_dict[img_id]
    pred_boxes = pred_box_dict[imgid2boxid[img_id]][:10]
    if include_null:
      caption = [NULL]+caption
    alignment = align_info['alignment']
    align_probs = None if not 'align_probs' in align_info else align_info['align_probs']

    # Append aligned captions to the predicted boxes
    # print(img_id, alignment, caption)
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

    for i_box, box in merged_boxes.items(): 
      pred_f.write('{} {} {} {} {} {}\n'.format(img_id, box[0], box[1], box[2], box[3], '_'.join(box[4]))) 
    merged_boxes_all.append(merged_boxes)
  return merged_boxes_all


def filter_segment_boxes(caption_file, 
                 pred_alignment_file, 
                 pred_box_file, 
                 gold_box_file,
                 out_file,
                 include_null=False,
                 keep_id_file=None):
  # Filter predicted boxes by merging boxes matched to the same true boxes
  # and create a predicted box file of the format:
  #   [image_id]_1 [x_min] [y_min] [x_max] [y_max]\
  #                [segment 1 start,segment 1 end]_..._[segment K start],[segment K end] (optional: [scores]) 
  # ------
  # Inputs:
  # ------
  #   caption_file: text file, each line containing a caption with format: 
  #                [segment_1] [segment_2]...[segment_n] 
  #   pred_alignment_file: json file storing the alignment
  #   pred_box_file: text file, each line containing a box with format:
  #                 [image_id] [label] [x] [y] [width] [height] 
  #   gold_box_file:  text file, each line containing a box with format:
  #                 [image_id] [label] [x] [y] [width] [height] 
  #   keep_id_file (optional): text file where example i is used if the i-th line is '1' and not used if the i-th
  #                            line is '0'
  # -------
  # Outputs:
  # -------
  #   merged_boxes: a list of boxes,
  #     [[[x_min_b, y_min_b, x_max_b, y_max_b, labels_b] for b in range(n_boxes_i)] for i in range(n)]
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
         
  with open(caption_file, 'r') as capt_f,\
       open(pred_alignment_file, 'r') as align_f:
    alignments = json.load(align_f)

    if keep_id_file:
      with open(keep_id_file, 'r') as keep_f:
        keep_indices = [ex for ex, line in enumerate(keep_f) if int(line)]
    
    if keep_id_file:
      captions = [line.strip().split() for ex, line in enumerate(capt_f) if ex in keep_indices]
    else:
      captions = [line.strip().split() for line in capt_f]
  
  # Load gold boxes
  gold_box_dict = {}
  prev_img_id = ''
  img_ids = []
  ex = -1
  with open(gold_box_file, 'r') as fg:
    for line_gold_box in fg:
      raw_gold_box_info = line_gold_box.strip().split()
      cur_img_id = raw_gold_box_info[0]
      x, y, w, h = raw_gold_box_info[-4:]
      x, y, w, h = int(x), int(y), int(w), int(h)
      
      # Convert the box format from [x, y, w, h] to 
      # [x_min, y_min, x_max, y_max]
      box_info = [x, y, x+w, y+h, raw_gold_box_info[1]]  
      if prev_img_id != cur_img_id:
        prev_img_id = cur_img_id
        ex += 1
        if not ex in keep_indices:
          continue
        # if ex >= 100: # XXX
        #   break
        gold_box_dict[cur_img_id] = [box_info]
        img_ids.append(cur_img_id)
        prev_img_id = cur_img_id 
      else:
        if not ex in keep_indices:
          continue
        gold_box_dict[cur_img_id].append(box_info)
  print('Number of gold boxes: {}'.format(len(gold_box_dict)))

  # Load predicted boxes
  if pred_box_file.split('.')[-1] == 'json':
    with open(pred_box_file, 'r') as fb:
      pred_box_dict = json.load(fb)
      pred_box_ids = sorted(pred_box_dict, key=lambda x:int(x.split('_')[-1]))
      imgid2boxid = {img_id:box_id for img_id in img_ids for box_id in pred_box_ids if '_'.join(box_id.split('_')[:-1]) == img_id}
  else:
    raise NotImplementedError('Predicted bbox file is assumed to be of json format for now')

  pred_f = open(out_file, 'w')
  merged_boxes_all = []
  for img_id, caption, align_info in zip(img_ids, captions, alignments):
    gold_boxes = gold_box_dict[img_id]
    pred_boxes = pred_box_dict[imgid2boxid[img_id]]
    if include_null:
        caption = [NULL_SEGMENT]+caption
    alignment = align_info['alignment']
    align_probs = None if not 'align_probs' in align_info else align_info['align_probs']
    
    start = 0
    # Append aligned segments to the predicted boxes
    for i_box, align_idx in enumerate(alignment):
        units = caption[align_idx].split(',') 
        pred_boxes[i_box].append('{},{}'.format(start, start+len(units)))
        start += len(units)
    
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

    for i_box, box in merged_boxes.items(): 
      pred_f.write('{} {} {} {} {} {}\n'.format(img_id, box[0], box[1], box[2], box[3], '_'.join(box[4]))) 
    merged_boxes_all.append(merged_boxes)
  return merged_boxes_all


def term_discovery_retrieval_metrics(pred_file,
                                     gold_file, 
                                     phone2idx_file=None, 
                                     tol=0, 
                                     visualize=False, 
                                     mode='iou', 
                                     out_file='scores'):
  # Calculate boundary F1 and token F1 scores from text files
  # Inputs:
  # ------
  #   pred_file: text file of the following format:
  #     Class 0:
  #     arr_0  [start time]  [end time]
  #     ...
  #     Class n:
  #     arr_0  [start time] [end time] 
  #     ...
  #   
  #   gold file: text file of the following format:
  #     arr_0 [start time] [end time] [token label]
  #     ...
  #     arr_N [start time] [end time] [token label]
  pred_boundaries, gold_boundaries = {}, {}
  pred_units, gold_units = {}, {}

  if phone2idx_file:
    with open(phone2idx_file, 'r') as f_i:
      phone2idx = json.load(f_i)
  else:
    with open(gold_file, 'r') as f_g:
      phone2idx = {}
      for line in f_g:
        k = line.strip().split()[-1]
        if not k in phone2idx: 
          phone2idx[k] = len(phone2idx)
  phone2idx[NULL] = len(phone2idx)
  phones = sorted(phone2idx, key=lambda x:phone2idx[x])
  n_phones = len(phone2idx)

  with open(pred_file, 'r') as f_p,\
       open(gold_file, 'r') as f_g:
    # Parse the discovered unit file
    class_idx = -1
    n_class = 0
    i = 0
    for line in f_p:
      # if i > 30: # XXX
      #   break
      # i += 1
      if line == '\n':
        continue
      if line.split()[0] == 'Class':
        class_idx = int(line.split(':')[0].split()[-1]) 
        n_class += 1
      else:
        example_id, start, end = line.split()
        start, end = float(start), float(end)
        if not example_id in pred_units:
          pred_boundaries[example_id] = [end]
          pred_units[example_id] = [class_idx]
        elif end > pred_boundaries[example_id][-1]:
          pred_boundaries[example_id].append(end)
          pred_units[example_id].append(class_idx)
        elif end < pred_boundaries[example_id][-1]:
          pred_boundaries[example_id].insert(0, end)
          pred_units[example_id].insert(0, class_idx)

    i = 0
    for line in f_g:
      # if i > 30: # XXX
      #   break
      # i += 1
      parts = line.strip().split()
      if len(parts) == 3:
        parts.append(NULL)
      example_id, start, end, phn = parts 
      
      if not phn in phone2idx:
        if phn in P2S:
          phn = P2S[phn]
        else:
          phn = NULL
      phn_idx = phone2idx[phn]
        
      start, end = float(start), float(end)      
      if not example_id in gold_boundaries:
        gold_boundaries[example_id] = [end]
        gold_units[example_id] = [phn_idx]
      elif end > gold_boundaries[example_id][-1]:
        gold_boundaries[example_id].append(end)
        gold_units[example_id].append(phn_idx)
      elif end < gold_boundaries[example_id][-1]:
        gold_boundaries[example_id].insert(0, end)
        gold_units[example_id].insert(0, phn_idx)
  print('Number of phone classes, number of phone clusters: {} {}'.format(n_phones, n_class))

  n = len(gold_boundaries)  
  n_gold_segments = 0.
  n_pred_segments = 0.
  n_correct_segments = 0.
  token_confusion = np.zeros((n_phones, n_class))

  print('Number of gold examples, number of predicted examples: {} {}'.format(len(gold_units), len(pred_units)))
  for i_ex, example_id in enumerate(sorted(gold_boundaries, key=lambda x:int(x.split('_')[-1]))):
    cur_gold_boundaries = gold_boundaries[example_id]
    n_gold_segments += len(cur_gold_boundaries)
    if cur_gold_boundaries[0] != 0:
      cur_gold_boundaries.insert(0, 0)
    cur_gold_units = gold_units[example_id]
    cur_pred_boundaries = pred_boundaries[example_id]
    n_pred_segments += len(cur_pred_boundaries)
    if cur_gold_boundaries[0] != 0:
      cur_pred_boundaries.insert(0, 0)
    cur_pred_units = pred_units[example_id]

    for gold_start, gold_end, gold_unit in zip(cur_gold_boundaries[:-1], cur_gold_boundaries[1:], cur_gold_units):      
      for pred_start, pred_end, pred_unit in zip(cur_pred_boundaries[:-1], cur_pred_boundaries[1:], cur_pred_units):       
        if abs(pred_end - gold_end) <= tol:
          n_correct_segments += 1.
          break

      found = 0
      for pred_start, pred_end, pred_unit in zip(cur_pred_boundaries[:-1], cur_pred_boundaries[1:], cur_pred_units):       
        if mode == 'iou':
          if (abs(pred_end - gold_end) <= tol and abs(pred_start - gold_start) <= tol) or IoU((pred_start, pred_end), (gold_start, gold_end)) > 0.5:
            found = 1
            break
        elif mode == 'hit':
          if pred_start >= gold_start and pred_end <= gold_end:
            found = 1
            break
        else:
          raise ValueError('Invalid mode {}'.format(mode))

      if found:
        token_confusion[gold_unit, pred_unit] += 1. 
  
  # print(n_correct_segments, n_gold_segments, n_pred_segments)
  boundary_rec = n_correct_segments / n_gold_segments
  boundary_prec = n_correct_segments / n_pred_segments
  if boundary_rec <= 0. or boundary_prec <= 0.:
    boundary_f1 = 0.
  else:
    boundary_f1 = 2. / (1. / boundary_rec + 1. / boundary_prec)

  token_recs = np.max(token_confusion, axis=0) / np.maximum(np.sum(token_confusion, axis=0), 1.)
  majority_classes = np.argmax(token_confusion, axis=0)
  token_precs = np.max(token_confusion, axis=1) / np.maximum(np.sum(token_confusion, axis=1), 1.)
  token_rec = np.mean(token_recs)
  token_prec = np.mean(token_precs)
  if token_rec <= 0. or token_prec <= 0.:
    token_f1 = 0.
  else:
    token_f1 = 2. / (1. / token_rec + 1. / token_prec)
 
  if out_file: 
    with open('{}.txt'.format(out_file), 'w') as f:
      f.write('Boundary recall: {}\n'.format(boundary_rec))
      f.write('Boundary precision: {}\n'.format(boundary_prec))
      f.write('Boundary f1: {}\n'.format(boundary_f1))
      f.write('Token recall: {}\n'.format(token_rec))
      f.write('Token precision: {}\n'.format(token_prec))
      f.write('Token f1: {}\n'.format(token_f1)) 
      sorted_indices = np.argsort(-token_recs)
      token_recs, majority_classes = token_recs[sorted_indices], majority_classes[sorted_indices]
      for c, (purity, label) in enumerate(zip(token_recs, majority_classes)):
        f.write('Cluster {} purity: {}, majority class: {}\n'.format(c, purity, phones[label]))
      
    print('Boundary recall: {}, precision: {}, f1: {}'.format(boundary_rec, boundary_prec, boundary_f1))
    print('Token recall: {}, precision: {}, f1: {}'.format(token_rec, token_prec, token_f1))
    return boundary_f1, token_f1

  if visualize:
    fig, ax = plt.subplots(figsize=(25, 25))
    token_confusion /= np.maximum(np.sum(token_confusion, axis=1, keepdims=True), 1.)
    best_classes = np.argmax(token_confusion, axis=1)
    print(len(best_classes))

    ax.set_yticks(np.arange(n_phones)+0.5, minor=False)
    ax.invert_yaxis()
    ax.set_yticklabels([phn for phn in sorted(phone2idx, key=lambda x:phone2idx[x])], minor=False)
    for tick in ax.get_yticklabels():
      tick.set_fontsize(25)

    ax.set_xticks(np.arange(n_phones)+0.5, minor=False)
    ax.set_xticklabels([str(c) for c in best_classes])
    for tick in ax.get_xticklabels():
      tick.set_fontsize(25)
      tick.set_rotation(90)
    
    plt.pcolor(token_confusion[:, best_classes], cmap=plt.cm.Greys, vmin=0, vmax=1)
    cbar = plt.colorbar()
    for tick in cbar.ax.get_yticklabels():
      tick.set_fontsize(50)

    plt.savefig('{}.png'.format(out_file), dpi=100)
    plt.close()

def find_segment(xs, ys):
  if len(xs) > len(ys):
    return -1
  
  for t in range(len(ys)):
    mismatch = np.sum([x != ys[t+l] for l, x in enumerate(xs) if t+l < len(ys)])
    if not mismatch:
      return t
  return -1

if __name__ == '__main__':
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--exp_dir', '-e', type=str, default='/ws/ifp-53_2/hasegawa/lwang114/fall2020/exp/cont_mixture_mscoco_davenet_rcnn_10region_topwordonly_10_16_2020/')
  parser.add_argument('--data_dir', '-d', default='/ws/ifp-53_2/hasegawa/lwang114/data/mscoco/') 
  parser.add_argument('--include_null', action='store_true')
  parser.add_argument('--segment', action='store_true', help='Assume word boundaries are unknown')
  args = parser.parse_args()

  caption_file = '{}/val2014/mscoco_val_text_captions.txt'.format(args.data_dir)
  unfiltered_pred_box_file = '{}/val2014/mscoco_val_bboxes_rcnn.json'.format(args.data_dir)
  gold_box_file = '{}/val2014/mscoco_val_bboxes.txt'.format(args.data_dir)
  keep_id_file = '{}/val2014/mscoco_val_split.txt'.format(args.data_dir)

  pred_alignment_file = '{}/alignment.json'.format(args.exp_dir)
  # if not os.path.isfile(pred_file):
  if not args.segment:
      pred_file = '{}/predicted_annotated_boxes.txt'.format(args.exp_dir)
      gold_file = '{}/gold_annotated_boxes.txt'.format(args.exp_dir)
      _ = filter_boxes(caption_file, 
                       pred_alignment_file, 
                       unfiltered_pred_box_file,
                       gold_box_file,
                       out_file=pred_file,
                       keep_id_file=keep_id_file)
      create_gold_annotated_box_file(caption_file, 
                                     gold_box_file, 
                                     out_file=gold_file, 
                                     keep_id_file=keep_id_file) # XXX
      recall, precision, f1 = alignmentF1(pred_file, gold_file)
      result_file = '{}/alignment_results.txt'.format(args.exp_dir)
  # if not os.path.isfile(gold_file):
  else:
      pred_file = '{}/predicted_segment_annotated_boxes.txt'.format(args.exp_dir)
      gold_file = '{}/gold_segment_annotated_boxes.txt'.format(args.exp_dir)
      _ = filter_boxes(caption_file, 
                       pred_alignment_file, 
                       unfiltered_pred_box_file,
                       gold_box_file,
                       out_file=pred_file,
                       keep_id_file=keep_id_file)
      create_gold_annotated_segment_box_file(caption_file, 
                                             gold_box_file, 
                                             out_file=gold_file, 
                                             keep_id_file=keep_id_file) # XXX
      recall, precision, f1 = segmentAlignmentF1(pred_file, gold_file)
      result_file = '{}/segment_alignment_results.txt'.format(args.exp_dir)

  with open(result_file, 'w') as f:
    print('Alignment recall={:.2f}, precision={:.2f}, f1={:.2f}'.format(recall * 100, precision * 100, f1 * 100))
    f.write('Alignment recall={:.2f}, precision={:.2f}, f1={:.2f}'.format(recall * 100, precision * 100, f1 * 100))
