import numpy as np
import json
import os
import argparse
import pkg_resources 
from tde.readers.gold_reader import *
from tde.readers.disc_reader import *
from tde.measures.grouping import * 
from tde.measures.coverage import *
from tde.measures.boundary import *
from tde.measures.ned import *
from tde.measures.token_type import *

P2S = {'men':'man', 'women':'woman', 'boys':'boy', 'girls':'girl', 'children':'child'}
def compute_cluster_labels(feat_file, 
                           codebook_file): # TODO
  pass

def attention_to_alignment(attention_file, 
                           image_concept_file,
                           out_file): # TODO Generate alignment without image concept_file

  with open(attention_file, 'r') as att_f,\
       open(image_concept_file, 'r') as concept_f,\
       open('{}_alignment.json'.format(out_file), 'w') as align_f:
    attention_dict = json.load(att_f)
    image_concepts = json.load(concept_f)

    align_dicts = []
    for ex, (k, attention) in enumerate(sorted(attention_dict.items(), key=lambda x:int(x[0].split('_')[-1]))):
      image_concept = image_concepts[ex]
      alignment = np.argmax(attention, axis=0)
      align_dicts.append({'alignment': alignment,
                        'image_concept': image_concept})
    json.dump(align_dicts, align_f)
  
def alignment_to_word_units(alignment_file, 
                            caption_file,
                            class2idx_file,
                            out_file):
  PERSON_S_CLASS = ['man', 'woman', 'boy', 'girl', 'child']

  # Read caption corpus, alignments and concept cluster labels 
  captions = []
  concepts = []
  with open(caption_file, 'r') as capt_f,\
       open(class2idx_file, 'r') as class2idx_f,\
       open(alignment_file, 'r') as align_f,\
       open('{}_class2idx.json'.format(out_file), 'w') as new_class_f:
    for line in capt_f:
      captions.append(line.strip().split())
    
    class2idx = json.load(class2idx_f)
    for c in PERSON_S_CLASS:
      class2idx[c] = len(class2idx) 
    class2idx['not_visual'] = len(class2idx)
    json.dump(class2idx, new_class_f, indent=4, sort_keys=True)
    alignments = json.load(align_f)

  captions = captions[::3] # XXX
  
  with open('{}.wrd'.format(out_file), 'w') as word_f,\
       open('{}.phn'.format(out_file), 'w') as phone_f,\
       open('{}_discovered_words.class'.format(out_file), 'w') as pred_f:
      # Create gold clusters
      gold_units = {}
      for ex, caption in enumerate(captions): # XXX
        # print('Caption {}'.format(ex))
        for t, w in enumerate(caption):
          if not w in class2idx and not w in P2S:
            continue
          if 'arr_{}'.format(ex) in gold_units:
            gold_units['arr_{}'.format(ex)].append((t, t+1, w))
          else:
            gold_units['arr_{}'.format(ex)] = [(t, t+1, w)]
          phone_f.write('arr_{} {} {} {}\n'.format(ex, t, t+1, w))
          word_f.write('arr_{} {} {} {}\n'.format(ex, t, t+1, w))

      # Create predicted clusters by selecting word units that align to the concept clusters
      pred_units = {}
      print('len(gold_units), len(captions), len(alignments): {} {} {}'.format(len(gold_units), len(captions), len(alignments)))
      for k in sorted(gold_units, key=lambda x:int(x.split('_')[-1])):
        # print('Alignment {}'.format(ex))
        ex = int(k.split('_')[-1])
        caption = captions[ex]
        align_info = alignments[ex]

        alignment = align_info['alignment']
        concepts = align_info['image_concepts']
        concept_assignment = {} 
        for align_idx, c in zip(alignment, concepts): # XXX Assume image is the target language
          if not str(align_idx) in concept_assignment:
            concept_assignment[str(align_idx)] = [c]
          else:
            concept_assignment[str(align_idx)].append(c)

        for token in gold_units[k]:
          t = str(token[0])
          if not t in concept_assignment:
            c_best = class2idx['not_visual']
          elif len(concept_assignment[t]) >= 1: # TODO Use translation probability to decide the assignment instead of majority vote; Merge consecutive phones that align to the same concept 
            i_best = np.argmax([concept_assignment[t].count(c) for c in concept_assignment[t]])
            c_best = concept_assignment[t][i_best]
            
          if str(c_best) in pred_units:
            pred_units[str(c_best)].append('{} {} {}'.format(k, token[0], token[1]))
          else:
            pred_units[str(c_best)] = ['{} {} {}'.format(k, token[0], token[1])] 
            
      for c in sorted(pred_units, key=lambda x:int(x)):
        pred_f.write('Class {}:\n'.format(c))
        pred_f.write('\n'.join(pred_units[c]))
        pred_f.write('\n\n')
          
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
      pred_f.write('\n'.join(pred_units[c]))
      pred_f.write('\n')

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
      
def term_discovery_retrieval_metrics(pred_file, gold_file, phone2idx_file=None, tol=0, visualize=False, out_file='scores'):
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
        phn = P2S[phn]
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
  n_gold_segments = 0
  n_pred_segments = 0
  n_correct_segments = 0
  token_confusion = np.zeros((n_phones, n_class))
  print(token_confusion.shape)
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
        if abs(pred_end - gold_end) <= tol:
          n_correct_segments += 1
          break

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

      
if __name__ == '__main__':
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--exp_dir', '-e', type=str, default='/ws/ifp-53_2/hasegawa/lwang114/fall2020/exp/cont_mixture_mscoco_davenet_rcnn_10_7_2020')
  args = parser.parse_args()

  alignment_file = '{}/alignment.json'.format(args.exp_dir)
  data_root = '/ws/ifp-53_2/hasegawa/lwang114/data/mscoco/'
  
  caption_file = '{}/train2014/mscoco_train_text_captions.txt'.format(data_root)
  class2idx_file = '{}/concept2idx_65class.json'.format(data_root)

  alignment_to_word_units(alignment_file, 
                          caption_file, 
                          class2idx_file,
                          out_file='{}/mscoco'.format(args.exp_dir))

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
  term_discovery_retrieval_metrics('{}/mscoco_discovered_words.class'.format(args.exp_dir),
                                   '{}/mscoco.wrd'.format(args.exp_dir),
                                   phone2idx_file='{}/mscoco_class2idx.json'.format(args.exp_dir),
                                   out_file='{}/tde_results'.format(args.exp_dir))
