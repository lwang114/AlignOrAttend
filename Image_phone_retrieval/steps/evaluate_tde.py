#!/usr/bin/env python

import numpy as np
import json
import os
import argparse
import pkg_resources 
import logging
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
'''
from tde.readers.gold_reader import *
from tde.readers.disc_reader import *
from tde.measures.grouping import * 
from tde.measures.coverage import *
from tde.measures.boundary import *
from tde.measures.ned import *
from tde.measures.token_type import *
'''

PUNCT = [',', '\'', '\"', '/', '?', '>', '<', '#', '%', '&', '*', ':', ';', '!', '.']
STOP = stopwords.words('english') + ['er', 'oh', 'ah', 'uh', 'um', 'ha']
PERSON_S_CLASS = ['man', 'woman', 'boy', 'girl', 'child']
P2S = {'men':'man', 'women':'woman', 'boys':'boy', 'girls':'girl', 'children':'child'}
NULL = '<NULL>'
def load_segmented_captions(capt_file, dataset='flickr'):
    captions = []
    segmentations = []
    # Load captions and segmentation
    if dataset == 'flickr':
      with open(capt_file, 'r') as capt_f:
          for line_capt in capt_f:
              caption = line_capt.strip().split()
              captions.append(caption)

              start = 0
              segmentation = []
              for w in caption:
                  dur = len(w.split(','))
                  segmentation.append([start, start+dur])
                  start += dur
              segmentations.append(segmentation)
    elif dataset == 'speechcoco':
      prev_start_frame, prev_end_frame = -1, -1 # XXX
      with open(capt_file, 'r') as capt_f:
        captions = []
        cur_capt_id = ''
        for line_capt in capt_f:
          capt_id, word, start, end = line_capt.strip().split() # TODO Confirm format
          start_frame, end_frame = int(float(start) / 10), int(float(end) / 10)
          if capt_id != cur_capt_id:
            captions.append([word])
            segmentations.append([(start_frame, end_frame)])
            cur_capt_id = capt_id
            prev_start_frame = start_frame # XXX
            prev_end_frame = end_frame # XXX
          else:
            if prev_end_frame > 0 and start_frame != prev_end_frame:
              segmentations[-1].append((prev_end_frame, start_frame)) # XXX Temporary: include the time gap between consecutive words
              captions[-1].append(NULL)
            captions[-1].append(word)
            segmentations[-1].append((start_frame, end_frame))
            prev_start_frame = start_frame # XXX
            prev_end_frame = end_frame # XXX

    return captions, segmentations

def load_boxes(box_file, 
               pron_file=None, 
               top_word_file=None,
               box_type='gold',
               dataset = 'flickr',
               out_file=None):
    lemmatizer = WordNetLemmatizer()
    if top_word_file and pron_file:
        with open(top_word_file, 'r') as top_f,\
             open(pron_file, 'r') as pron_f:
            top_words = json.load(top_f)
            pron_dict = json.load(pron_f)
    elif box_type == 'gold' and dataset == 'flickr':
        raise ValueError('Top word list and pronunciation dictionary are required for gold boxes')

    cur_img_key = ''
    boxes = []
    img_keys = [] 
    class2idx = {}
    ex = 0 
    box_f = open(box_file, 'r')
    for line_box in box_f:
      raw_box_info = line_box.strip().split()
      img_key = raw_box_info[0]

      assert box_type in {'gold', 'pred'}
      if box_type == 'gold':      
        if dataset == 'speechcoco':
          x, y, w, h = raw_box_info[-4:]
          x, y, w, h = float(x), float(y), float(w), float(h)
          box_info = [x, y, x+w, y+h, raw_box_info[1]]
          if not raw_box_info[1] in class2idx:
            class2idx[raw_box_info[1]] = len(class2idx)
        else: 
          box_info = [float(x) for x in raw_box_info[-4:]]
        # TODO Extract the last nouns as label of the region; if no noun exists, choose the whole phrase
        '''
        pos_tags = nltk.pos_tag(raw_box_info[1:-4])
        found = 0
        for label, tag in pos_tags[::-1]:
            if tag[0] == 'N':
                label = lemmatizer.lemmatize(label.lower())
                if (not label in top_words) or (label in STOP+PUNCT): 
                    continue
                else:
                    found = 1
                    break
        if found:
            c = ','.join(pron_dict[label])
            box_info.append(c)
        else:
        '''
        if dataset == 'flickr':
          for label in raw_box_info[1:-4]:
            if (not label in top_words) or (label in STOP+PUNCT): 
              continue
            c = ','.join(pron_dict[label])
            box_info.append(c)
          class2idx['_'.join(box_info[4:])] = len(class2idx)

      elif box_type == 'pred':
        if dataset == 'flickr':
          box_info = [float(x) for x in raw_box_info[2:6]]

      if img_key != cur_img_key:
        ex += 1
        # if ex >= 500: # XXX
        #   break
        boxes.append([box_info])
        img_keys.append(img_key)
        cur_img_key = img_key
        # print('Number of images={}, number of images kept={}, image key={}'.format(len(img_keys), ex, img_key))
      else:
        boxes[-1].append(box_info)
    box_f.close()
    if out_file:
      class2idx_f = open('{}_class2idx.json'.format(out_file), 'w')
      json.dump(class2idx, class2idx_f, indent=4, sort_keys=True)
      class2idx_f.close()
    return boxes, img_keys

def match_boxes(pred_boxes, gold_boxes):
  # For each predicted box, find the gold box with which it overlaps the most;
  # Use the phrase of the gold box as its phrase
  #
  # Inputs:
  # ------
  #   pred_boxes: [(x^i, y^i, w^i, h^i, label^i_1,..., label^i_L) for i in range(len(pred_boxes))] 
  #   gold_boxes: [(x^i, y^i, w^i, h^i, label^i_1,..., label^i_L) for i in range(len(gold_boxes))] 
  #
  # Outputs:
  # -------
  #   units: [(j_i, x^{j_i}, y^{j_i}, w^{j_i}, h^{j_i}, label^{j_i}_1,..., label^{j_i})_L) for i in range(len(pred_boxes))]
  units = []
  gold_labels = [' '.join(gbox[4:]) for gbox in gold_boxes]
  for pbox in pred_boxes:
    ious = []
    for gbox in gold_boxes:
      # print(pbox[:4], gbox[:4])
      ious.append(IoU(pbox[:4], gbox[:4]))

    i_best = np.argmax(ious)
    units.append([i_best]+gold_boxes[i_best])
  return units

      
def match_caption_with_boxes(caption, boxes):
  # Inputs:
  # ------
  #   caption: [[w for w in sent] for sent in corpus]
  #   boxes: [(x^i, y^i, w^i, h^i, label^i_1,..., label^i_l) for i in range(len(boxes))]
  #
  # Outputs:
  # -------
  #   units: [(start_{t_i}, end_{t_i}, label_{t_i}) for i in range(len(boxes))]
  units = []
  for i_box, box in enumerate(boxes):
    if len(box) < 5:
      box.append(NULL)
    labels = box[4:]
    start = find_segment(labels, caption)
    if start == -1:
      print('Warning: string {} not found in {}'.format(labels, caption))
    else:
      n_phones = 0
      for w in box[4:]:
        n_phones += len(w.split(','))
      units.append([i_box, start, start+n_phones, '_'.join(labels)])
  return units

 
def speechcoco_extract_gold_units(caption_file, 
                                  box_file,
                                  class2idx_file=None,
                                  out_file='speechcoco'):
  # Read caption corpus, alignments and concept cluster labels 
  captions = []
  concepts = []
  with open(caption_file, 'r') as capt_f:
    for line in capt_f:
      captions.append(line.strip().split()) 
    ''' 
    class2idx = json.load(class2idx_f)
    for c in PERSON_S_CLASS:
      class2idx[c] = len(class2idx) 
    class2idx['not_visual'] = len(class2idx)
    json.dump(class2idx, new_class_f, indent=4, sort_keys=True)
    '''
  
  captions, segmentations = load_segmented_captions(caption_file, dataset='speechcoco')
  captions = captions[::3] # XXX
  segmentations = segmentations[::3]
  boxes, img_ids = load_boxes(box_file, dataset='speechcoco', out_file=out_file)
  boxes = boxes[::3]
  img_ids = img_ids[::3]
  ignore_ids = []
  print('Number of captions={}, number of segmentations={}, number of boxes={}'.format(len(captions), len(segmentations), len(boxes)))

  with open('{}.wrd'.format(out_file), 'w') as word_f,\
       open('{}.phn'.format(out_file), 'w') as phone_f,\
       open('{}.link'.format(out_file), 'w') as link_f:
      # Create gold clusters
      ex = -1
      for img_id, caption, segmentation, box in zip(img_ids, captions, segmentations, boxes):
        caption = [w if not w in PERSON_S_CLASS else 'person' for w in caption]
        caption = [w if not w in P2S else 'person' for w in caption]
        units = match_caption_with_boxes(caption, box) # TODO Consider a broader person class
        if len(units) == 0:
            ignore_ids.append(img_id)
            continue
        ex += 1
        # if not w in class2idx and not w in P2S:
        #   continue
        # if 'arr_{}'.format(ex) in gold_units:
        # print('Caption {}'.format(ex))

        for seg, unit in zip(segmentation, units):
          start, end = seg
          phone_f.write('{}_{} {} {} {}\n'.format(img_id, ex, start, end, unit[3]))
          word_f.write('{}_{} {} {} {}\n'.format(img_id, ex, start, end, unit[3]))
          link_f.write('{}_{} {} {} {}\n'.format(img_id, ex, unit[1], start, end))
  return ignore_ids
          
def speechcoco_extract_pred_units(pred_alignment_file,
                                  segment_file,
                                  pred_box_file, 
                                  gold_box_file,
                                  ignore_ids=None,
                                  out_file='speechcoco'):
  with open(alignment_file, 'r') as align_f,\
       open('{}_discovered_words.class'.format(out_file), 'w') as pred_f,\
       open('{}_discovered_links.txt'.format(out_file), 'w') as pred_link_f:
    # Create predicted clusters by selecting word units that align to the concept clusters
    alignments = json.load(align_f)
    gold_boxes, img_ids = load_boxes(gold_box_file, dataset='speechcoco')
    pred_boxes = json.load(open(pred_box_file, 'r'))
    gold_boxes = gold_boxes[::3] # XXX
    img_ids = img_ids[::3]
    pred_boxes = [pred_boxes[img_id] for img_id in sorted(pred_boxes, key=lambda x:int(x.split('_')[-1])) if '_'.join(img_id.split('_')[:-1]) in img_ids]
    captions, segmentations = load_segmented_captions(segment_file, dataset='speechcoco') 
    captions = captions[::3] # XXX
    segmentations = segmentations[::3]

    print('Number of images with predicted boxes={}, number of images with gold boxes={}'.format(len(pred_boxes), len(gold_boxes)))
    print('Number of captions={}, number of segmentations={}'.format(len(captions), len(segmentations)))

    pred_units = {}
    pred_links = []
    ex = -1
    for img_id, pred_box, gold_box in zip(img_ids, pred_boxes, gold_boxes):
        if img_id in ignore_ids:
            continue
        ex += 1
        alignment = alignments[ex]['alignment']
        segmentation = segmentations[ex]
        box_alignment = match_boxes(pred_box, gold_box)

        for i_pred_box, align_idx in enumerate(alignment):
          i_gold_box = box_alignment[i_pred_box][0]
          gold_box = box_alignment[i_pred_box][1:]
          segment = segmentation[align_idx]
          label = gold_box[4]

          if not label in pred_units:
            pred_units[label] = ['{}_{} {} {}'.format(img_id, ex, segment[0], segment[1])]
          else:
            pred_units[label].append('{}_{} {} {}'.format(img_id, ex, segment[0], segment[1]))
          pred_links.append('{}_{} {} {} {}'.format(img_id, ex, i_gold_box, segment[0], segment[1]))

    print('Number of examples kept={}'.format(ex+1))      
         
    for i_label, label in enumerate(pred_units):
      pred_f.write('Class {}:\n'.format(i_label))
      pred_f.write('\n'.join(list(set(pred_units[label])))) # Set removes repetitive units
      pred_f.write('\n\n')
    pred_link_f.write('\n'.join(list(set(pred_links)))) # Set removes repetitive links

    

def flickr_extract_gold_units(caption_file,
                              box_file,
                              pron_file,
                              top_word_file='/ws/ifp-53_2/hasegawa/lwang114/data/flickr30k/flickr30k_word_to_idx_filtered.json',
                              ignore_ids_file='/ws/ifp-53_2/hasegawa/lwang114/data/flickr30k/flickr8k_test_ids.txt',
                              out_file='flickr'):
  lemmatizer = WordNetLemmatizer()
  with open(caption_file, 'r') as capt_f,\
       open(ignore_ids_file, 'r') as ignore_f,\
       open(pron_file, 'r') as pron_f,\
       open(top_word_file, 'r') as top_f,\
       open('{}.wrd'.format(out_file), 'w') as wrd_f,\
       open('{}.phn'.format(out_file), 'w') as phn_f,\
       open('{}.link'.format(out_file), 'w') as link_f:
      pron_dict = json.load(pron_f)
      top_words = json.load(top_f)
      ignore_ids = ['_'.join(line.split('_')[:-1]) for line in ignore_f]

      captions = []
      for line in capt_f:
        caption = []
        for w in line.split():
            w = lemmatizer.lemmatize(w.lower())
            if (not w in top_words) or (w in STOP+PUNCT):
                continue
            caption.append(','.join(pron_dict[w]))
        captions.append(caption)

      boxes, img_ids = load_boxes(box_file, pron_file, top_word_file, out_file=out_file)
      print('Number of boxes: {}, number of image ids: {}, number of captions: {}'.format(len(boxes), len(img_ids), len(captions)))
          
      ex = -1
      for img_id, caption, box in zip(img_ids, captions, boxes):
        # print('Example {}, img_id {}'.format(ex, img_id))
        # if ex > 100: # XXX
        #   break
        if img_id.split('.')[0] in ignore_ids:
          continue
        ex += 1
        units = match_caption_with_boxes(caption, box)
        for unit in units:
          # print(unit)
          i_box = unit[0]
          start = unit[1]
          end = unit[2]
          link_f.write('{}_{} {} {} {}\n'.format(img_id, ex, i_box, start, end))
          wrd_f.write('{}_{} {} {} {}\n'.format(img_id, ex, start, end, unit[3].encode('utf-8')))
          
          for wrd in unit[3].split('_'):
            for phn in wrd.split(','): 
              phn_f.write('{}_{} {} {} {}\n'.format(img_id, ex, start, start+1, phn.encode('utf-8')))
              start += 1
  print('Number of examples={}'.format(len(boxes)))


def flickr_extract_pred_units(pred_alignment_file,
                              segmented_caption_file,
                              pred_box_file, 
                              gold_box_file,
                              pron_file,
                              top_word_file='/ws/ifp-53_2/hasegawa/lwang114/data/flickr30k/flickr30k_word_to_idx_filtered.json',
                              ignore_ids_file=None,
                              out_file='flickr',
                              include_null=True): 
  with open(pred_alignment_file, 'r') as pred_align_f,\
       open(ignore_ids_file, 'r') as ignore_f,\
       open('{}_discovered_words.class'.format(out_file), 'w') as pred_f,\
       open('{}_discovered_links.txt'.format(out_file), 'w') as pred_link_f:

    ignore_ids = ['_'.join(line.split('_')[:-1]) for line in ignore_f]
    alignments = json.load(pred_align_f)

    gold_boxes, img_ids = load_boxes(gold_box_file, pron_file, top_word_file)
    pred_boxes, _ = load_boxes(pred_box_file, box_type='pred')
    captions, segmentations = load_segmented_captions(segmented_caption_file)
    print('Number of images with predicted boxes={}, number of images with gold boxes={}'.format(len(pred_boxes), len(gold_boxes)))
    print('Number of captions={}, number of segmentations={}'.format(len(captions), len(segmentations)))

    pred_units = {}
    pred_links = []
    ex = -1
    for img_id, gold_box, pred_box, segmentation, caption in zip(img_ids, gold_boxes, pred_boxes, segmentations, captions):
        if img_id.split('.')[0] in ignore_ids:
            continue
        ex += 1
        alignment = alignments[ex]['alignment']
        box_alignment = match_boxes(pred_box, gold_box)

        for i_pred_box, align_idx in enumerate(alignment): 
          # print('i_pred_box, len(box_alignment): {} {}'.format(i_pred_box, len(box_alignment)))
          # print('len(caption), align_idx: {} {}'.format(len(caption), align_idx))
          # print('len(alignment): {}'.format(len(alignment)))
          # print(caption)
          # print(box_alignment)
          i_gold_box = box_alignment[i_pred_box][0]
          gold_box = box_alignment[i_pred_box][1:]
          segment = segmentation[align_idx-1] if include_null else segmentation[align_idx] 
          label = '_'.join(gold_box[4:])

          if not label in pred_units:
            pred_units[label] = ['{}_{} {} {}'.format(img_id, ex, segment[0], segment[1])]
          else:
            pred_units[label].append('{}_{} {} {}'.format(img_id, ex, segment[0], segment[1]))
          pred_links.append('{}_{} {} {} {}'.format(img_id, ex, i_gold_box, segment[0], segment[1]))
    print('Number of examples kept={}'.format(ex+1))

    for i_label, label in enumerate(pred_units):
        pred_f.write('Class {}\n'.format(i_label))
        pred_f.write('\n'.join(list(set(pred_units[label])))) # set removes repetitive segments
        pred_f.write('\n\n')
    pred_link_f.write('\n'.join(list(set(pred_links))))

def cluster_to_word_units(cluster_file,
                          out_file):
  pred_units = {}
  with open(cluster_file, 'r') as cluster_f,\
       open('{}_discovered_words.class'.format(out_file), 'w') as pred_f:
    cluster_dict = json.load(cluster_f)
    for k, cluster_labels in sorted(cluster_dict.items(), key=lambda x:int(x[0].split('_')[-1])):
      for t, c in enumerate(cluster_labels):
        pred_units[c].append('{} {} {}'.format(k, t, t+1))

    for c in sorted(pred_units, key=lambda x:int(x)):
      pred_f.write('Class {}:\n'.format(c))
      pred_f.write('\n'.join(list(set(pred_units[c]))))
      pred_f.write('\n')

def compute_cluster_labels(feat_file, 
                           codebook_file): # TODO
  pass
 
def IoU(pred, gold):
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

def find_segment(xs, ys):
  if len(xs) > len(ys):
    return -1
  
  for t in range(len(ys)):
    mismatch = np.sum([x != ys[t+l] for l, x in enumerate(xs) if t+l < len(ys)])
    if not mismatch:
      return t
  return -1

def term_discovery_retrieval_metrics(pred_file, gold_file, phone2idx_file=None, tol=0, visualize=False, mode='iou', out_file='scores'):
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
      
    print('Boundary recall: {}, precision: {}, f1: {}\n'.format(boundary_rec, boundary_prec, boundary_f1))
    print('Token recall: {}, precision: {}, f1: {}\n'.format(token_rec, token_prec, token_f1))
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


def linkF1(pred_file, gold_file, out_file):
    correct = 0
    n_pred_links = 0
    n_gold_links = 0
    with open(pred_file, 'r') as pred_f,\
         open(gold_file, 'r') as gold_f,\
         open(out_file, 'w') as out_f:
        pred_links = []
        gold_links = []
        cur_img_id = ''
        for line in pred_f:
            parts = line.split()
            img_id = parts[0]
            link = (int(parts[1]), float(parts[2]), float(parts[3]))
            if not img_id == cur_img_id: 
                pred_links.append([link])
                cur_img_id = img_id
            else:
                pred_links[-1].append(link)

        cur_img_id = ''
        for line in gold_f:
            parts = line.split()
            img_id = parts[0]
            link = (int(parts[1]), float(parts[2]), float(parts[3]))
            if not img_id == cur_img_id: 
                gold_links.append([link])
                cur_img_id = img_id
            else:
                gold_links[-1].append(link)

        print('Number of gold examples, pred examples: {} {}'.format(len(gold_links), len(pred_links)))
        for cur_pred_links, cur_gold_links in zip(pred_links, gold_links):
            n_gold_links += len(cur_gold_links)
            n_pred_links += len(cur_pred_links)
            for plink in cur_pred_links:
                ious = []
                for glink in cur_gold_links:
                    # if plink[0] == glink[0] and plink[1] > glink[1] and plink[2] < glink[2]:
                    ious.append(IoU(plink[1:], glink[1:]))    
                correct += max(ious)

        recall = correct / n_gold_links
        prec = correct / n_pred_links
        f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0
        print('Link recall: {:2f}, link precision: {:2f}, link F1: {:2f}'.format(recall*100, prec*100, f1*100))
        out_f.write('Link recall: {:2f}, link precision: {:2f}, link F1: {:2f}'.format(recall*100, prec*100, f1*100))
      
if __name__ == '__main__':
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--exp_dir', '-e', type=str, default='/ws/ifp-53_2/hasegawa/lwang114/fall2020/exp/cont_mixture_mscoco_davenet_rcnn_10_7_2020')
  parser.add_argument('--dataset', '-d', choices={'speechcoco', 'flickr', 'mscoco'}, default='speechcoco')  
  args = parser.parse_args()


  if args.dataset == 'speechcoco':
    alignment_file = '{}/alignment.json'.format(args.exp_dir)
    data_root = '/ws/ifp-53_2/hasegawa/lwang114/data/mscoco/'
    caption_file = '{}/train2014/mscoco_train_text_captions.txt'.format(data_root)
    segment_file = '{}/train2014/mscoco_train_word_segments.txt'.format(data_root)
    gold_box_file = '{}/train2014/mscoco_train_bboxes.txt'.format(data_root) 
    pred_box_file = '{}/train2014/mscoco_train_bboxes_rcnn.json'.format(data_root) 

    ignore_ids = speechcoco_extract_gold_units(segment_file, 
                                               gold_box_file,
                                               out_file='{}/speechcoco'.format(args.exp_dir))
    speechcoco_extract_pred_units(alignment_file,
                                  segment_file,
                                  pred_box_file,
                                  gold_box_file,
                                  ignore_ids=ignore_ids,
                                  out_file='{}/speechcoco'.format(args.exp_dir))
    linkF1('{}/speechcoco_discovered_links.txt'.format(args.exp_dir),
           '{}/speechcoco.link'.format(args.exp_dir),
           out_file='{}/link_results'.format(args.exp_dir)) 
    term_discovery_retrieval_metrics('{}/speechcoco_discovered_words.class'.format(args.exp_dir),
                                   '{}/speechcoco.wrd'.format(args.exp_dir),
                                   mode = 'hit',
                                   phone2idx_file='{}/speechcoco_class2idx.json'.format(args.exp_dir),
                                   out_file='{}/tde_results'.format(args.exp_dir))
  elif args.dataset == 'flickr':
    args.exp_dir = '/ws/ifp-53_2/hasegawa/lwang114/fall2020/exp/cont_mixture_aligner_flickr30k_phone_rcnn_10_1_2020' 
    alignment_file = '{}/alignment.json'.format(args.exp_dir)
    data_root = '/ws/ifp-53_2/hasegawa/lwang114/data/flickr30k/'
    text_caption_file = '{}/flickr30k_text_captions_filtered.txt'.format(data_root)
    segmented_caption_file = '{}/flickr30k_phone_captions_segmented.txt'.format(data_root)
    pron_file = '{}/flickr30k_word_pronunciations.json'.format(data_root)
    pred_box_file = '{}/flickr30k_rcnn_bboxes.txt'.format(data_root)
    gold_box_file = '{}/flickr30k_phrases_bboxes.txt'.format(data_root)
    test_ids_file = '{}/flickr8k_test.txt'.format(data_root)
    
    flickr_extract_gold_units(text_caption_file,
                              gold_box_file,
                              pron_file,
                              ignore_ids_file = test_ids_file,
                              out_file='{}/flickr'.format(args.exp_dir))
        
    flickr_extract_pred_units(alignment_file,
                              segmented_caption_file,
                              pred_box_file, 
                              gold_box_file,
                              pron_file,
                              ignore_ids_file = test_ids_file,
                              out_file='{}/flickr'.format(args.exp_dir))
    
    linkF1('{}/flickr_discovered_links.txt'.format(args.exp_dir),
           '{}/flickr.link'.format(args.exp_dir),
           out_file='{}/link_results'.format(args.exp_dir))
    term_discovery_retrieval_metrics('{}/flickr_discovered_words.class'.format(args.exp_dir),
                                     '{}/flickr.wrd'.format(args.exp_dir),
                                     mode = 'hit',
                                     out_file='{}/tde_results_hit'.format(args.exp_dir))
    term_discovery_retrieval_metrics('{}/flickr_discovered_words.class'.format(args.exp_dir),
                                     '{}/flickr.wrd'.format(args.exp_dir),
                                     mode = 'iou',
                                     out_file='{}/tde_results_iou'.format(args.exp_dir))

  '''
  wrd_path = pkg_resources.resource_filename(
    pkg_resources.Requirement.parse('tde'),
    '/tde/share/mscoco.wrd')
  phn_path = pkg_resources.resource_filename(
    pkg_resources.Requirement.parse('tde'),
    '/tde/share/mscoco.phn')
  gold = Gold(wrd_path=wrd_path,
              phn_path=phn_path)
  discovered = Disc('{}/mscoco_discovered_words.class'.format(args.exp_dir), gold)

  with open('{}/tde_results.txt'.format(args.exp_dir), 'w') as tde_f:
    coverage = Coverage(gold, discovered)
    coverage.compute_coverage()
    print('Coverage: ', coverage.coverage)
    tde_f.write('Coverage: {}\n'.format(coverage.coverage))
    
    token_type = TokenType(gold, discovered)
    token_type.compute_token_type()
    token_type.compute_token_type()
    token_f1 = (2 * token_type.precision[0] * token_type.recall[0]) / (token_type.precision[0] + token_type.recall[0])\
               if token_type.precision[0] + token_type.recall[0] > 0 else 0
    type_f1 = (2 * token_type.precision[1] * token_type.recall[1]) / (token_type.precision[1] + token_type.recall[1])\
              if token_type.precision[1] + token_type.recall[1] > 0 else 0
    print('Token recall: {}\tprecision: {}\tF1: {}'.format(token_type.recall[0], token_type.precision[0], token_f1))
    print('Type recall: {}\tprecision: {}\tF1: {}'.format(token_type.recall[1], token_type.precision[1], token_f1))
    tde_f.write('Token recall: {}\tprecision: {}\tF1: {}'.format(token_type.recall[0], token_type.precision[0], token_f1))
    tde_f.write('Type recall: {}\tprecision: {}\tF1: {}'.format(token_type.recall[1], token_type.precision[1], token_f1))
  '''
