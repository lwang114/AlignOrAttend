import numpy as np
import json
import os

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
  P2S = {'men':'man', 'women':'woman', 'boys':'boy', 'girls':'girl', 'children':'child'}
  # Read caption corpus, alignments and concept cluster labels 
  captions = []
  concepts = []
  with open(caption_file, 'r') as capt_f,
       open(class2idx_file, 'r') as class2idx_f,
       open(alignment_file, 'r') as align_f:
    for line in capt_f:
      captions.append(line.strip().split())
    
    class2idx = json.load(class2idx_f)
    for c in PERSON_S_CLASS:
      class2idx[c] = len(class2idx) 
    
    alignments = json.load(align_f)

  with open('{}.wrd'.format(out_file), 'w') as word_f,\
       open('{}.phn'.format(out_file), 'w') as phone_f,\
       open('{}_discovered_words.class', 'w') as pred_f:
      # Create gold clusters  
      for ex, caption in enumerate(captions):
        for t, w in enumerate(caption):
          phone_f.write('arr_{} {} {} {}\n'.format(ex, t, t+1, w))
        
          if not w in class2idx and not w in P2S:
            continue
          word_f.write('arr_{} {} {} {}\n'.format(ex, t, t+1, w))

      # Create predicted clusters by selecting word units that align to the concept clusters
      pred_units = {} 
      for ex, (caption, align_info) in enumerate(zip(captions, alignments)):
        alignment = align_info['alignment']
        concepts = align_info['image_concepts']
        concept_assignment = [[] for _ in alignment] 
        for align_idx, c in zip(alignment, concepts): # XXX Assume image is the target language
          concept_assignment[align_idx].append(c)

        for t in concept_assignment: 
          if len(concept_assignment[t]) > 1: # TODO Use translation probability to decide the assignment instead of majority vote; Merge consecutive phones that align to the same concept 
            i_best = np.argmax([concept_assignment.count(c) for c in concept_assignment])
            c_best = concept_assignment[i_best]
            if c_best in pred_units:
              pred_units[c_best].append('arr_{} {} {}'.format(ex, t, t+1))
            else:
              pred_units[c_best] = ['arr_{} {} {}'.format(ex, t, t+1)] 
      for c in sorted(pred_units, key=lambda x:int(x)):
        pred_f.write('Class {}:\n'.format(c))
        pred_f.write('\n'.join(pred_units[c]))
        pred_f.write('\n')
        
          
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

