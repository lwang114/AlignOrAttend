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
from sklearn.metrics import precision_recall_curve, average_precision_score 
import pandas as pd

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
NULL_SEGMENT = '0,0'

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
  labels = []
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
    n_gold += len([gold for gold in gold_alignments_i if gold[5] != NULL_SEGMENT])

    for pred in pred_alignments_i:
      n_pred += 1
      for gold in gold_alignments_i:
        if functools.reduce(lambda x, y: x and y, map(lambda g, p: g==p, gold[1:5], pred[1:5]), True):
          gold_label = gold[5]
          pred_labels = pred[5]
          if functools.reduce(lambda x, y: x or y, map(lambda p: _find_segment(p, gold_label), pred_labels), False):
            # print(gold, pred)
            correct += 1 
          else:
            pred_class_labels = [W2C[pred_label] for pred_label in pred_labels if pred_label in W2C]
            if functools.reduce(lambda x, y: x or y, map(lambda p: _find_segment(p, gold_label), pred_labels), False):
              correct += 1
  
  print('{} corrects, {} gold alignments, {} pred alignments'.format(correct, n_gold, n_pred))
  recall = correct / n_gold
  precision = correct / n_pred
  f1 = 2 * recall * precision / (recall + precision) if recall + precision > 0  else 0
  return recall, precision, f1

def alignment_average_precision(pred_file,
                                gold_file,
                                out_file):
  # ------
  # Inputs:
  # ------
  #   pred_file: text file of the following format: 
  #       [image_id]_1 [x_min] [y_min] [x_max] [y_max] [caption label] [score]
  #       ...
  #       [image_id]_N [x_min] [y_min] [x_max] [y_max] [caption label] [score]
  #
  #   gold_file: text file of the following format (having the same number of lines as pred_file):
  #       [image_id]_1 [x_min] [y_min] [x_max] [y_max] [caption label] [binary label]
  #       ...
  #       [image_id]_N [x_min] [y_min] [x_max] [y_max] [caption label] [binary label]
  # -------
  # Outputs:
  # -------
  #   precision, recall: lists of floats in the range [0, 1]
  #   thresholds: list of floats
  y_scores = []
  y_true = []
  labels = []
  with open(pred_file, 'r') as pred_f,\
       open(gold_file, 'r') as gold_f:
    y_scores = [float(line.strip().split()[6]) for line in pred_f]
    y_true = [int(line.strip().split()[6]) for line in gold_f]
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    print(precision, recall)
    auc_dict = {'precision': precision, 'recall': recall}
    auc_df = pd.DataFrame(auc_dict)
    auc_df.to_csv(out_file)
    average_precision = average_precision_score(y_true, y_scores)

  return average_precision

def retrieval_average_precision(pred_file,
                                out_file):
  # ------
  # Inputs:
  # ------
  #   pred_file: .npy file containing the predicted similarity score matrix
  # -------
  # Outputs:
  # -------
  #   precision, recall: lists of floats from [0, 1] for precision@n, recall@n respectively 
  #   thresholds: list of detection thresholds
  S = np.load(pred_file)
  n = S.shape[0]
  ranks = np.argsort(np.argsort(-S, axis=0), axis=0).T
  y_scores = -ranks.flatten(order='C')
  y_true = np.eye(n).flatten()
  precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
  ap_dict = {'precision': precision, 'recall': recall}
  ap_df = pd.DataFrame(ap_dict)
  ap_df.to_csv(out_file)
  average_precision = average_precision_score(y_true, y_scores)
  
  return average_precision 

def tokenF1(pred_file,
            gold_file):
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
  def _purity(pred_alignments, gold_alignments):
    confusion = {}  
    for pred_alignments_i, gold_alignments_i in zip(pred_alignments, gold_alignments):
      for pred in pred_alignments_i:
        for gold in gold_alignments_i:
          gold_label = gold[5]
          pred_label = pred[5]
          print(gold_label, pred_label)
          if functools.reduce(lambda x, y: x and y, map(lambda g, p: g==p, gold[1:5], pred[1:5]), True): 
            if not gold_label in confusion:
              confusion[gold_label] = {}
              confusion[gold_label][pred_label] = 1.
            elif not pred_label in confusion[gold_label]:
              confusion[gold_label][pred_label] = 1.
            else:
              confusion[gold_label][pred_label] += 1.  
          elif pred_label in W2C:
            pred_label = W2C[pred_label]
            if not gold_label in confusion:
              confusion[gold_label] = {}
              confusion[gold_label][pred_label] = 1.
            elif not pred_label in confusion[gold_label]:
              confusion[gold_label][pred_label] = 1.
            else:
              confusion[gold_label][pred_label] += 1. 

    purity = 0.  
    n_types = len(confusion)
    for g in confusion:
      n_gold = len(confusion[g].values())
      
      max_n = confusion[g][g] if g in confusion[g] else 0.
      purity +=  max_n / n_gold
    return purity / n_types

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
        for pred_label in pred_labels:
          pred_alignments.append([(img_id, x_min, y_min, x_max, y_max, pred_label)])
        cur_id = img_id
      else:
        for pred_label in pred_labels:
          pred_alignments[-1].append((img_id, x_min, y_min, x_max, y_max, pred_label))
    
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

  recall = _purity(pred_alignments, gold_alignments)
  precision = _purity(gold_alignments, pred_alignments)  
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
        # if ex >= 500: # XXX
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
  gold_annotated_boxes = []
  with open(caption_file, 'r') as capt_f,\
       open(box_file, 'r') as box_f:
    if keep_id_file: 
      with open(keep_id_file, 'r') as keep_f:
        keep_indices = [ex for ex, line in enumerate(keep_f) if int(line)]

    cur_id = ''
    captions = []
    for line_capt in capt_f:
      capt_id, word, start, end = line_capt.strip().split()
      start, end = int(start), int(end)
      if capt_id != cur_id:
        cur_id = capt_id
        captions.append([(capt_id, start, end, word)])
      else:
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
 

def create_alignment_label_file(caption_file,
                                box_file,
                                out_file='gold_alignment_label.txt',
                                keep_id_file=None):
  # Create a file containing the true alignments between words and gold boxes 
  # in the following format:
  #   [image_id]_1 [x_min] [y_min] [x_max] [y_max] [caption_label] [binary alignment label]
  # ------
  # Inputs:
  # ------
  #   caption_file: text file, each line containing a caption with format:
  #                 [word_1] [word_2]...[word_n]
  #   box_file: text file, each line containing a box with format:
  #             [image_id] [label] [x] [y] [width] [height]
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
        for t, word in enumerate(captions[ex]):
          if class_label == word:
            label = word
            gold_annotated_boxes.append((img_id, x, y, x+w, y+h, label, 1))
          elif class_label in C2W and word in C2W[class_label]:
            label = word
            gold_annotated_boxes.append((img_id, x, y, x+w, y+h, label, 1))
          elif class_label == 'person' and (word in PERSON_S_CLASS):
            label = word
            gold_annotated_boxes.append((img_id, x, y, x+w, y+h, label, 1))
          elif word in P2S:
            word_s = P2S[word]
            if word_s in PERSON_S_CLASS or word_s in C2W:
              label = word
              gold_annotated_boxes.append((img_id, x, y, x+w, y+h, label, 1))
            else:
              gold_annotated_boxes.append((img_id, x, y, x+w, y+h, label, 0))
          else:
            gold_annotated_boxes.append((img_id, x, y, x+w, y+h, label, 0))
        # print(ex, captions[ex], class_label, label) # XXX
      else:
        if not ex in keep_indices:
          continue
        label = NULL
        for word in captions[ex]:
          if class_label == word:
            label = word 
            gold_annotated_boxes.append((img_id, x, y, x+w, y+h, label, 1))
          elif class_label in C2W and word in C2W[class_label]:
            label = word
            gold_annotated_boxes.append((img_id, x, y, x+w, y+h, label, 1))
          elif class_label == 'person' and (word in PERSON_S_CLASS):
            label = word
            gold_annotated_boxes.append((img_id, x, y, x+w, y+h, label, 1))
          elif word in P2S:
            word_s = P2S[word]
            if word_s in PERSON_S_CLASS or word_s in C2W:
              label = word
              gold_annotated_boxes.append((img_id, x, y, x+w, y+h, label, 1))
            else:
              gold_annotated_boxes.append((img_id, x, y, x+w, y+h, label, 0))
          # print(ex, captions[ex], class_label, label) # XXX
          else:
            gold_annotated_boxes.append((img_id, x, y, x+w, y+h, label, 0))
  
  with open(out_file, 'w') as out_f:
    for gold_box in gold_annotated_boxes:
      out_f.write('{} {} {} {} {} {} {}\n'.format(gold_box[0],\
                                                  gold_box[1],\
                                                  gold_box[2],\
                                                  gold_box[3],\
                                                  gold_box[4],\
                                                  gold_box[5],\
                                                  gold_box[6]))

 

def create_alignment_score_file(caption_file,
                          pred_alignment_file,
                          pred_box_file,
                          gold_box_file,
                          out_file,
                          include_null=False,
                          keep_id_file=None):
  # Create two text files containing gold and predicted box-word alignment of the format:
  #   [image_id]_1 [x_min] [y_min] [x_max] [y_max] [caption_label] [score]
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
       open(pred_alignment_file, 'r') as align_f: # Find the alignments between pred boxes and gold boxes
    alignments = json.load(align_f)

    if keep_id_file:
      with open(keep_id_file, 'r') as keep_f:
        keep_indices = [ex for ex, line in enumerate(keep_f) if int(line)]
    
    if keep_id_file:
      captions = [line.strip().split() for ex, line in enumerate(capt_f) if ex in keep_indices]
    else:
      captions = [line.strip().split() for line in capt_f]
  
  gold_box_dict = {}
  prev_img_id = ''
  img_ids = []
  ex = -1
  with open(gold_box_file, 'r') as fg: # Load gold boxes
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

  if pred_box_file.split('.')[-1] == 'json': 
    with open(pred_box_file, 'r') as fb: # Load predicted boxes

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
    align_probs = np.asarray(align_info['align_probs'])
    # print(img_id, alignment, caption)
    for i_box, align_idx in enumerate(alignment):
      pred_boxes[i_box].append(caption[align_idx])
    box_alignment = _match_boxes(pred_boxes, gold_boxes) 

    # Find the predicted alignments between words and gold boxes from pred boxes
    goldbox2alignprobs = {}
    for i_pred, i_gold in enumerate(box_alignment):
      if not str(i_gold) in goldbox2alignprobs: 
        goldbox2alignprobs[str(i_gold)] = [align_probs[i_pred]] 
      else:
        goldbox2alignprobs[str(i_gold)].append(align_probs[i_pred])
      
    goldbox2alignprobs = {str(i_gold):np.asarray(goldbox2alignprobs[str(i_gold)]).mean(axis=0) for i_gold in goldbox2alignprobs}
    for i_gold, gold_box in enumerate(gold_boxes):
      if not str(i_gold) in goldbox2alignprobs:
        for t, word in enumerate(caption):
          pred_f.write('{} {} {} {} {} {} {}\n'.format(img_id, gold_box[1], gold_box[2], gold_box[3], gold_box[4], word, -1e14))
      else:
        for t, word in enumerate(caption):
          pred_f.write('{} {} {} {} {} {} {}\n'.format(img_id, gold_box[1], gold_box[2], gold_box[3], gold_box[4], word, goldbox2alignprobs[str(i_gold)][t]))  


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
        # if ex >= 500: # XXX
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
      if align_probs is not None:
        scores = [align_probs[i_pred_box] for i_pred_box, i_gold_box in enumerate(box_alignment) if i_gold_box == int(i_box)] 
        pred_f.write('{} {} {} {} {} {} {}\n'.format(img_id, box[0], box[1], box[2], box[3], '_'.join(box[4]), np.max(scores)))
      else:
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


if __name__ == '__main__':
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--exp_dir', '-e', type=str, default='/ws/ifp-53_2/hasegawa/lwang114/fall2020/exp/cont_mixture_mscoco_davenet_rcnn_10region_topwordonly_10_16_2020/')
  parser.add_argument('--data_dir', '-d', default='/ws/ifp-53_2/hasegawa/lwang114/data/mscoco/') 
  parser.add_argument('--include_null', action='store_true')
  parser.add_argument('--segment', action='store_true', help='Assume word boundaries are unknown')
  parser.add_argument('--task', '-t', type=int, default=0)
  args = parser.parse_args()

  unfiltered_pred_box_file = '{}/val2014/mscoco_val_bboxes_rcnn.json'.format(args.data_dir)
  gold_box_file = '{}/val2014/mscoco_val_bboxes.txt'.format(args.data_dir)
  keep_id_file = '{}/val2014/mscoco_val_split.txt'.format(args.data_dir)

  if args.task == 0:
    alignment_files = []
    for fn in os.listdir(args.exp_dir):
      if 'alignment' in fn and 'json' in fn and not 'train' in fn:
        alignment_files.append('{}/{}'.format(args.exp_dir, fn))
        
    # if not os.path.isfile(pred_file):
    result_file = '{}/alignment_results.txt'.format(args.exp_dir)
    res_f = open(result_file, 'w')
    if not args.segment:
        caption_file = '{}/val2014/mscoco_val_text_captions.txt'.format(args.data_dir)
        pred_file = '{}/predicted_annotated_boxes.txt'.format(args.exp_dir)
        gold_file = '{}/gold_annotated_boxes.txt'.format(args.exp_dir)
        for pred_alignment_file in alignment_files:
          print(pred_alignment_file)
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
          print('{}, alignment recall={:.2f}, precision={:.2f}, f1={:.2f}\n'.format(pred_alignment_file, recall * 100, precision * 100, f1 * 100))
          res_f.write('{}, alignment recall={:.2f}, precision={:.2f}, f1={:.2f}\n'.format(pred_alignment_file, recall * 100, precision * 100, f1 * 100))
    # if not os.path.isfile(gold_file):
    else:
        caption_file = '{}/val2014/mscoco_val_phone_captions_segmented.txt'.format(args.data_dir)
        segment_file = '{}/val2014/mscoco_val_word_phone_segments.txt'.format(args.data_dir)
        pred_file = '{}/predicted_segment_annotated_boxes.txt'.format(args.exp_dir)
        gold_file = '{}/gold_segment_annotated_boxes.txt'.format(args.exp_dir)
        for pred_alignment_file in alignment_files:
          _ = filter_segment_boxes(caption_file, 
                                   pred_alignment_file, 
                                   unfiltered_pred_box_file,
                                   gold_box_file,
                                   out_file=pred_file,
                                   keep_id_file=keep_id_file)
          create_gold_annotated_segment_box_file(segment_file, 
                                                 gold_box_file, 
                                                 out_file=gold_file, 
                                                 keep_id_file=keep_id_file) # XXX
          recall, precision, f1 = segmentAlignmentF1(pred_file, gold_file)
          print('{}, alignment recall={:.2f}, precision={:.2f}, f1={:.2f}\n'.format(pred_alignment_file, recall * 100, precision * 100, f1 * 100))
          res_f.write('{}, alignment recall={:.2f}, precision={:.2f}, f1={:.2f}\n'.format(pred_alignment_file, recall * 100, precision * 100, f1 * 100))
  elif args.task == 1:
    result_file = '{}/token_results.txt'.format(args.exp_dir)
    res_f = open(result_file, 'w')
    caption_file = '{}/val2014/mscoco_val_text_captions.txt'.format(args.data_dir)
    pred_file = os.path.join(args.exp_dir, 'predicted_annotated_boxes.txt')
    gold_file = os.path.join(args.exp_dir, 'gold_annotated_boxes.txt')
    pred_alignment_file = os.path.join(args.exp_dir, 'alignment.json')
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
    recall, precision, f1 = tokenF1(pred_file, gold_file)
    print('{}, token recall={:.2f}, precision={:.2f}, f1={:.2f}\n'.format(pred_alignment_file, recall * 100, precision * 100, f1 * 100))
    res_f.write('{}, token recall={:.2f}, precision={:.2f}, f1={:.2f}\n'.format(pred_alignment_file, recall * 100, precision * 100, f1 * 100))
  elif args.task == 2:
    result_file = '{}/AP_results.txt'.format(args.exp_dir)
    res_f = open(result_file, 'w')
    caption_file = '{}/val2014/mscoco_val_text_captions.txt'.format(args.data_dir)
    pred_file = os.path.join(args.exp_dir, 'binary_alignment_scores.txt')
    gold_file = os.path.join(args.exp_dir, 'binary_alignment_labels.txt')
    pred_alignment_file = os.path.join(args.exp_dir, 'alignment.json')
    create_alignment_label_file(caption_file,
                                gold_box_file,
                                out_file=gold_file,
                                keep_id_file=keep_id_file)
    create_alignment_score_file(caption_file, 
                                pred_alignment_file, 
                                unfiltered_pred_box_file,
                                gold_box_file,
                                out_file=pred_file,
                                keep_id_file=keep_id_file)
    ap = alignment_average_precision(pred_file, gold_file, '{}/avg_precision.csv'.format(args.exp_dir))
    print('Average precision={}\n'.format(ap))
    res_f.write('Average precision={}\n'.format(ap))
  elif args.task == 3:
    result_file = '{}/retrieval_AP_results.txt'.format(args.exp_dir)
    res_f = open(result_file, 'w') 
    pred_file = os.path.join(args.exp_dir, 'similarity.npy')
    ap = retrieval_average_precision(pred_file, '{}/retrieval_avg_precision.csv'.format(args.exp_dir))
    print('Average precision={}\n'.format(ap))
