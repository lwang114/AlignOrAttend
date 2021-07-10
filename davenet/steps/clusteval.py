import numpy as np
import json
from sklearn.metrics import roc_curve 
import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import argparse

DEBUG = False
NULL = 'NULL'
END = '</s>'
EPS = 1e-17

#logging.basicConfig(filename="clusteval.log", format="%(asctime)s %(message)s", level=logging.DEBUG)
#logging.basicConfig(format="%(asctime)s %(message)s", level=logging.DEBUG)

#
# Parameters:
# ----------  
# pred (clusters) --- A list of cluster indices corresponding to each sample
# gold (classes) --- A list of class indices corresponding to each sample
def cluster_confusion_matrix(pred, gold, create_plot=True, alignment=None, file_prefix='cluster_confusion_matrix'):
  assert len(pred) == len(gold) 
  n = len(pred)
  if alignment is not None:
    for i, a in enumerate(alignment):
      pred[i] = np.array(pred[i])[a['alignment']].tolist()

  n_cp, n_cg = 1, 1
  for p, g in zip(pred, gold):
    # XXX: Assume p, q are lists and class indices are zero-based
    for c_p, c_g in zip(p, g):
      if c_g + 1 > n_cg:
        n_cg = c_g + 1
      
      if c_p + 1 > n_cp:
        n_cp = c_p + 1
  cm = np.zeros((n_cp, n_cg))
  
  for p, g in zip(pred, gold):
    for c_p, c_g in zip(p, g):
      cm[c_p, c_g] += 1.

  cm = (cm / np.maximum(np.sum(cm, axis=0), EPS)).T
  print('Cluster purity: ', np.mean(np.max(cm, axis=-1)))
  if create_plot:
    fig, ax = plt.subplots(figsize=(20, 30))
    ax.set_xticks(np.arange(cm.shape[1])+0.5, minor=False)
    ax.set_yticks(np.arange(cm.shape[0])+0.5, minor=False)
    ax.set_xticklabels([str(c) for c in range(n_cg)], minor=False)
    ax.set_yticklabels([str(c) for c in range(n_cp)], minor=False) 
    plt.pcolor(cm, cmap=plt.cm.Blues, vmin=0, vmax=1)
    plt.savefig(file_prefix, dpi=100)
    plt.close()
  np.savez(file_prefix+'.npz', cm)

def word_cluster_confusion_matrix(pred_info, gold_info, concept2idx=None, file_prefix='audio_confusion_matrix'):
  pred, gold = [], []
  for p, g in zip(pred_info, gold_info):
    pred_assignment = p['concept_alignment']
    gold_alignment = g['alignment']

    i_prev = gold_alignment[0]
    if concept2idx is not None:
      concepts = [concept2idx[c] for c in g['image_concepts']]     
    else:
      concepts = g['image_concepts']

    gold_words = [concepts[i_prev]]
    gold_segmentations = [0]
    # Find the true concept label for each segment
    for i in gold_alignment:
      if i != i_prev:
        gold_words.append(concepts[i])
        gold_segmentations.append(i)
        i_prev = i
    gold.append(gold_words)

    pred_words = []
    for start, end in zip(gold_segmentations[:-1], gold_segmentations[1:]):
      segment = pred_assignment[start:end] 
      counts = {c:0 for c in list(set(segment))}
      for c in segment:
        counts[c] += 1
      pred_words.append(sorted(counts, key=lambda x:counts[x], reverse=True)[0])
    pred.append(pred_words)
  cluster_confusion_matrix(pred, gold) 

#
# Parameters:
# ----------  
# pred (clusters) --- A list of cluster indices corresponding to each sample
# gold (classes) --- A list of class indices corresponding to each sample
def cluster_purity(pred, gold):
  cp = 0.
  n = 0.

  cm = np.zeros
  for p, g in enumerate(pred, gold):
    n_intersects = [] 
    n_intersects.append(len(set(p).intersection(set(g))))

    cp += max(n_intersects)
    n += len(set(p))

  return cp / n

def alignment_retrieval_metrics(pred, gold, out_file='class_retrieval_scores.txt', max_len=2000, return_results=False, debug=False, print_results=True):
  assert len(pred) == len(gold)
  n = len(pred)
  prec = 0.
  rec = 0.

  v = max([max(g['alignment']) for g in gold]) + 1
  confusion = np.zeros((v, v))
  # Local retrieval metrics
  for n_ex, (p, g) in enumerate(zip(pred, gold)):
    p_ali = p['alignment'][:max_len]
    g_ali = g['alignment'][:max_len]
   
    # if debug:
    #   print("examples " + str(n_ex)) 
    #   print("# of frames in predicted alignment and gold alignment: %d %d" % (len(p_ali), len(g_ali))) 
    # XXX assert len(p_ali) == len(g_ali)
    
    for a_p, a_g in zip(p_ali, g_ali):
      confusion[a_g, a_p] += 1.
  f1 = v / np.sum(1. / np.maximum(np.diag(confusion) / np.maximum(np.sum(confusion, axis=1), 1), EPS))
  # if print_results: 
  #  print('Alignment F1: ' + str(f1))
  if return_results:
    return confusion

def term_discovery_retrieval_metrics(pred_file, gold_file, phone2idx_file=None, tol=3, visualize=False, out_file='scores'):
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

    if phone2idx_file:
      with open(phone2idx_file, 'r') as f_i:
        phone2idx = json.load(f_i)
        phones = [phn for phn in sorted(phone2idx, key=lambda x:phone2idx[x])]
        n_phones = len(phone2idx) 
    else:
      phone2idx = {}
      phones = []
      n_phones = 0

    i = 0
    for line in f_g:
      # if i > 30: # XXX
      #   break
      # i += 1
      example_id, start, end, phn = line.split()
      if not phn in phone2idx:
        if phone2idx_file:
          class_idx = int(phn)
          phn = phones[int(phn)]
        else:
          phone2idx[phn] = n_phones
          phones.append(phn)
          n_phones += 1
          class_idx = n_phones
      else:
        class_idx = phone2idx[phn]
      start, end = float(start), float(end)        
      if not example_id in gold_boundaries:
        gold_boundaries[example_id] = [end]
        gold_units[example_id] = [class_idx]
      elif end > gold_boundaries[example_id][-1]:
        gold_boundaries[example_id].append(end)
        gold_units[example_id].append(class_idx)
      elif end < gold_boundaries[example_id][-1]:
        gold_boundaries[example_id].insert(0, end)
        gold_units[example_id].insert(0, class_idx)
    print('Number of phone classes, number of phone clusters: {} {}'.format(n_phones, n_class))

  n = len(gold_boundaries)  
  n_gold_segments = 0
  n_pred_segments = 0
  n_correct_segments = 0
  token_confusion = np.zeros((n_phones, n_class))
  for i_ex, example_id in enumerate(sorted(gold_boundaries, key=lambda x:int(x.split('_')[-1]))):
    # print("Example %d" % i_ex)
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
        if i_ex == 0:
          print(pred_end, gold_end)
        if abs(pred_end - gold_end) <= tol:
          n_correct_segments += 1
          break

      if i_ex == 0:
        print('Final: ', pred_end, gold_end)

      found = 0
      for pred_start, pred_end, pred_unit in zip(cur_pred_boundaries[:-1], cur_pred_boundaries[1:], cur_pred_units):       
        if (abs(pred_end - gold_end) <= tol and abs(pred_start - gold_start) <= tol) or IoU((pred_start, pred_end), (gold_start, gold_end)) > 0.5:
          found = 1
          break
      if found:
        token_confusion[gold_unit, pred_unit] += 1          
  
  print(n_correct_segments, n_gold_segments, n_pred_segments)
  boundary_rec = n_correct_segments / n_gold_segments
  boundary_prec = n_correct_segments / n_pred_segments
  if boundary_rec <= 0. or boundary_prec <= 0.:
    boundary_f1 = 0.
  else:
    boundary_f1 = 2. / (1. / boundary_rec + 1. / boundary_prec)

  token_rec = np.mean(np.max(token_confusion, axis=0) / np.maximum(np.sum(token_confusion, axis=0), 1.))
  token_prec = np.mean(np.max(token_confusion, axis=1) / np.maximum(np.sum(token_confusion, axis=1), 1.))
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
  else:
    print('Boundary f1: {}\n'.format(boundary_f1))
    print('Token f1: {}\n'.format(token_f1))
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

def alignment_accuracy(pred, gold, pred_lm=None, gold_lm=None, max_len=2000):
  assert len(pred) == len(gold)
  acc = 0.
  n = 0.
  lm_keys = sorted(pred_lm, key=lambda x:int(x.split('_')[-1]))
  for i_ex, (p, g) in enumerate(zip(pred, gold)):
    ali_p = p['alignment'][:max_len]
    lm_p = pred_lm[lm_keys[i_ex]]
    if lm_p[0] != 0:
      lm_p = np.append(0, lm_p)

    ali_g = g['alignment'][:max_len]
    lm_g = gold_lm[lm_keys[i_ex]]
    if lm_g[0] != 0:
      lm_g = np.append(0, lm_g)
    lm_g = gold_lm[lm_keys[i_ex]]
    
    if pred_lm and gold_lm:
      ali_p = [a_p for a_p, start, end in zip(ali_p, lm_p[:-1], lm_p[1:]) for _ in range(end-start)]
      ali_g = [a_g for a_g, start, end in zip(ali_g, lm_g[:-1], lm_g[1:]) for _ in range(end-start)] 
    for a_p, a_g in zip(ali_p, ali_g):
      acc += (a_p == a_g)
      n += 1
  
  return acc / n

def word_IoU(pred, gold): 
  if DEBUG:
    logging.debug("# of examples in pred and gold: %d %d" % (len(pred), len(gold)))
  assert len(pred) == len(gold)
  iou = 0.
  n = 0.
  for p, g in zip(pred, gold):
    p_word_boundaries = _findWords(p['alignment'])
    g_word_boundaries = _findWords(g['alignment'])
    
    if DEBUG:
      logging.debug("pred word boundaries: " + str(p_word_boundaries))
      logging.debug("groundtruth word boundaries: " + str(g_word_boundaries))
    
    for p_wb in p_word_boundaries: 
      n_overlaps = []
      for g_wb in g_word_boundaries:
        n_overlaps.append(IoU(g_wb, p_wb))
      max_iou = max(n_overlaps)
      iou += max_iou
      n += 1
  return iou / n

def IoU(pred, gold):
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
 
def _findWords(alignment):
  cur = alignment[0]
  start = 0
  boundaries = []
  for i, a_i in enumerate(alignment):
    if a_i != cur:
      boundaries.append((start, i))
      start = i
      cur = a_i
    if DEBUG:
      print(i, a_i, start, cur)

  boundaries.append((start, len(alignment)))
  return boundaries

if __name__ == '__main__':
  tasks = [2]
  parser = argparse.ArgumentParser()
  parser.add_argument('--exp_dir', '-e', type=str, default='./', help='Experiment Directory')
  parser.add_argument('--dataset', '-d', choices=['flickr', 'flickr_audio', 'mscoco2k', 'mscoco20k'], help='Dataset')
  parser.add_argument('--tolerance', '-t', type=float, default=3, help='Tolerance for boundary F1')
  args = parser.parse_args()
  if args.dataset == 'flickr':
    gold_json = '../data/flickr30k/phoneme_level/flickr30k_gold_alignment.json'
    concept2idx_file = '../data/flickr30k/concept2idx.json'
  elif args.dataset == 'flickr_audio':
    gold_json = '../data/flickr30k/audio_level/flickr30k_gold_alignment.json'
    concept2idx_file = '../data/flickr30k/concept2idx.json'
  elif args.dataset == 'mscoco2k' or args.dataset == 'mscoco20k':
    gold_json = '../data/mscoco/%s_gold_alignment.json' % args.dataset
    concept2idx_file = '../data/mscoco/concept2idx_integer.json'
  else:
    raise ValueError('Dataset not specified or not valid')
  #----------------------------#
  # Create Majority Prediction #
  #----------------------------#
  if 0 in tasks:
    with open(args.exp_dir+'model_names.txt', 'r') as f:
      model_names = f.read().strip().split()
    
    if args.dataset == 'mscoco2k' or args.dataset == 'mscoco20k':
      with open('../data/mscoco/concept2idx_integer.json', 'w') as f:
        json.dump({i:i for i in range(65)}, f, indent=4, sort_keys=True)
    
    with open(gold_json, 'r') as f:
      gold_dict = json.load(f)
    with open(concept2idx_file, 'r') as f:
      concept2idx = json.load(f)
 
    majority_dict = []
    for g in gold_dict:
      p = {}
      p['image_concepts'] = [concept2idx[c] for c in g['image_concepts']]
      p['alignment'] = [0]*len(g['alignment'])
      p['index'] = g['index']
      majority_dict.append(p)
    
    with open('%s_%s_pred_alignment.json' % (args.exp_dir + args.dataset, 'majority'), 'w') as f:
      json.dump(majority_dict, f, indent=4, sort_keys=True)
  #-------------------#
  # Alignment Metrics #
  #-------------------#
  if 1 in tasks:
    if args.dataset == 'mscoco2k' or args.dataset == 'mscoco20k':
      with open('../data/mscoco/concept2idx_integer.json', 'w') as f:
        json.dump({i:i for i in range(65)}, f, indent=4, sort_keys=True)

    with open(concept2idx_file, 'r') as f:
      concept2idx = json.load(f) 

    with open(args.exp_dir+'model_names.txt', 'r') as f:
      model_names = f.read().strip().split()
    
    for model_name in model_names:
      pred_json = '%s_%s_pred_alignment.json' % (args.exp_dir + args.dataset, model_name) 
      with open(pred_json, 'r') as f:   
        pred_dict = json.load(f)

      with open(gold_json, 'r') as f:
        gold_dict = json.load(f)
            
      with open(concept2idx_file, 'r') as f:
        concept2idx = json.load(f)

      pred = []
      gold = []
      for p, g in zip(pred_dict, gold_dict):
        pred.append(p['image_concepts'])
        gold.append([concept2idx[str(c)] for c in g['image_concepts']])
 
      print('Accuracy: ', accuracy(pred_dict, gold_dict))
      alignment_retrieval_metrics(pred_dict, gold_dict, debug=False)
  #------------------------#
  # Term Discovery Metrics #
  #------------------------#
  if 2 in tasks:
    # Try using gold landmarks to generate pred_file and evaluate it using gold_file
    landmark_file = '/ws/ifp-04_3/hasegawa/lwang114/spring2020/data/mscoco2k_wbd_landmarks.npz' 
    landmark_dict = np.load(landmark_file)
    gold_file = '{}/{}'.format(args.exp_dir, 'mscoco2k_word_units.wrd') # '/home/lwang114/spring2019/MultimodalWordDiscovery/utils/tdev2/WDE/share/mscoco2k_phone_units.phn'
    phone2idx_file = '/ws/ifp-53_2/hasegawa/lwang114/data/mscoco/concept2idx_65class.json' # '/ws/ifp-53_2/hasegawa/lwang114/data/mscoco/mscoco_phone2id.json'

    with open('{}/{}'.format(args.exp_dir, 'model_names.txt'), 'r') as f:
      model_names = f.read().strip().split()

    for model_name in model_names:
      print(model_name)
      pred_file = '{}/discovered_words_{}.class'.format(args.exp_dir, model_name) # '/home/lwang114/spring2019/MultimodalWordDiscovery/utils/tdev2/WDE/share/discovered_words_{}.class'.format(model_name)     
      term_discovery_retrieval_metrics(pred_file, gold_file, phone2idx_file=phone2idx_file, tol=args.tolerance, out_file='{}/{}'.format(args.exp_dir, model_name), visualize=True) 
