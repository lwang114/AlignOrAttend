#-----------------------------------------------------------------------------------# 
#                           CONTINUOUS MIXTURE ALIGNER CLASS                        #
#-----------------------------------------------------------------------------------# 
# Author: Liming Wang
# Email: limingwanggrant@gmail.com 
import numpy as np
import sklearn
from sklearn.cluster import KMeans
from sklearn.mixture import BayesianGaussianMixture
import argparse
import logging
import os
import json
from region_vgmm import *
import torch
from NegativeSquare import NegativeSquare

logger = logging.getLogger(__name__)
EPS = 1e-15
class FullyContinuousMixtureAligner(object):
  """An alignment model based on Brown et. al., 1993. capable of modeling continuous bilingual sentences"""
  def __init__(self, source_features_train, target_features_train, configs):
    self.Ks = configs.get('n_src_vocab', 80)
    self.Kt = configs.get('n_trg_vocab', 2001)
    self.use_null = configs.get('use_null', True)
    self.pretrained_vgmm_model = configs.get('pretrained_vgmm_model', None)
    self.pretrained_translateprob = configs.get('pretrained_translateprob', None)
    var = configs.get('var', 160.) # XXX
    logger.info('n_src_vocab={}, n_trg_vocab={}'.format(self.Ks, self.Kt))
    self.alpha = configs.get('alpha', 0.)
    if target_features_train[0].ndim <= 1:
      self.trg_embedding_dim = 1 
    else:
      self.trg_embedding_dim = target_features_train[0].shape[-1]
    print('target embedding dimension={}'.format(self.trg_embedding_dim))

    self.src_vec_ids_train = []
    start_index = 0
    for ex, src_feat in enumerate(source_features_train):
      if self.use_null:        
        if self.trg_embedding_dim == 1:
          target_features_train[ex] = [self.Kt-1]+target_features_train[ex]
      src_vec_ids = []
      for t in range(len(src_feat)):
        src_vec_ids.append(start_index+t)
      start_index += len(src_feat)
      self.src_vec_ids_train.append(src_vec_ids)
    
    self.src_model = RegionVGMM(np.concatenate(source_features_train, axis=0),
                                self.Ks,
                                var=var,
                                vec_ids=self.src_vec_ids_train,
                                pretrained_model=self.pretrained_vgmm_model)
    self.src_feats = self.src_model.X
    self.trg_feats = target_features_train
    if self.pretrained_translateprob:
      self.P_ts = np.load(self.pretrained_translateprob)
      print('Loaded pretrained translation probabilities')
    else:
      self.P_ts = 1./self.Ks * np.ones((self.Kt, self.Ks))
    self.trg2src_counts = np.zeros((self.Kt, self.Ks))

  def compute_forward_probs(self, src_sent, trg_sent):
    T = src_sent.shape[0]
    L = trg_sent.shape[0]
    A = np.ones((L, L)) / max(L, 1)
    init = np.ones(L) / max(L, 1)
    forward_probs = np.zeros((T, L, self.Kt))
    scales = np.zeros((T,))
    
    probs_x_t_given_z = src_sent @ self.P_ts.T
    forward_probs[0] = np.tile(init[:, np.newaxis], (1, self.Kt)) * trg_sent * probs_x_t_given_z[0] 
    scales[0] = np.sum(forward_probs[0])
    forward_probs[0] /= np.maximum(scales[0], EPS)
    A_diag = np.diag(np.diag(A))
    A_offdiag = A - A_diag
    
    for t in range(T-1):
      probs_x_t_z_given_y = trg_sent * probs_x_t_given_z[t+1]
      forward_probs[t+1] += (A_diag @ forward_probs[t]) * probs_x_t_given_z[t+1]
      forward_probs[t+1] += ((A_offdiag.T @ np.sum(forward_probs[t], axis=-1)) * probs_x_t_z_given_y.T).T
      scales[t+1] = np.sum(forward_probs[t+1])
      forward_probs[t+1] /= max(scales[t+1], EPS)
    return forward_probs, scales
      
  def compute_backward_probs(self, src_sent, trg_sent, scales):
    T = src_sent.shape[0]
    L = trg_sent.shape[0]
    A = np.ones((L, L)) / max(L, 1)
    init = np.ones(L) / max(L, 1)
    backward_probs = np.zeros((T, L, self.Kt))
    backward_probs[T-1] = 1.

    A_diag = np.diag(np.diag(A))
    A_offdiag = A - A_diag
    probs_x_t_given_z = src_sent @ self.P_ts.T
    
    for t in range(T-1, 0, -1):
      probs_x_t_z_given_y = trg_sent * probs_x_t_given_z[t]
      backward_probs[t-1] = A_diag @ (backward_probs[t] * probs_x_t_given_z[t])
      backward_probs[t-1] += np.tile(A_offdiag @ np.sum(backward_probs[t] * probs_x_t_z_given_y, axis=-1)[:, np.newaxis], (1, self.Kt))
      backward_probs[t-1] /= max(scales[t], EPS) 
    return backward_probs
    
  def update_counts(self):
    # Update alignment counts
    log_probs = []
    self.trg2src_counts[:] = 0.
    for i, (trg_feat, src_vec_ids) in enumerate(zip(self.trg_feats, self.src_vec_ids_train)):
      src_feat = self.src_feats[src_vec_ids]
      C_ts, log_prob_i = self.update_counts_i(i, src_feat, trg_feat)
      self.trg2src_counts += C_ts
      log_probs.append(log_prob_i)

    self.P_ts = deepcopy(self.translate_prob())
    return np.mean(log_probs)

  def update_counts_i(self, i, src_feat, trg_feat):
    src_sent = np.exp(self.src_model.log_prob_z(i, normalize=False))
    trg_sent = trg_feat

    V_src = to_one_hot(src_sent, self.Ks)
    V_trg = to_one_hot(trg_sent, self.Kt)
   
    forward_probs, scales = self.compute_forward_probs(V_src, V_trg)
    backward_probs = self.compute_backward_probs(V_src, V_trg, scales)
    print('forward_probs * backward_probs: ', (forward_probs * backward_probs).sum(axis=(1, 2))) # XXX
    norm_factor = np.sum(forward_probs * backward_probs, axis=(1, 2), keepdims=True)
    new_state_counts = forward_probs * backward_probs / np.maximum(norm_factor, EPS)
    C_ts = np.sum(new_state_counts, axis=1).T @ (V_src / np.maximum(np.sum(V_src, axis=1, keepdims=True), EPS))
    log_prob = np.log(np.maximum(scales, EPS)).sum()
    return C_ts, log_prob

  def update_components(self):
    means_new = np.zeros(self.src_model.means.shape)
    counts = np.zeros((self.Ks,))
    for i, (trg_feat, src_feat) in enumerate(zip(self.trg_feats, self.src_feats)):
      if len(trg_feat) == 0 or len(self.src_feats[i]) == 0:
        continue 
      trg_sent = trg_feat
      # (src len, src vocab size)
      prob_f_given_y = self.prob_s_given_tsent(trg_sent)
      # (src len, src vocab size)
      prob_f_given_x = np.exp(self.src_model.log_prob_z(i))
      post_f = prob_f_given_y * prob_f_given_x
      post_f /= np.maximum(np.sum(post_f, axis=1, keepdims=True), EPS)
  
      # Update target word counts of the target model
      indices = self.src_vec_ids_train[i]
     
      means_new += np.sum(post_f[:, :, np.newaxis] * self.src_model.X[indices, np.newaxis], axis=0)
      counts += np.sum(post_f, axis=0)
      # self.update_components_exact(i, ws=post_f, method='exact') 
    self.src_model.means = deepcopy(means_new / np.maximum(counts[:, np.newaxis], EPS)) 
     
  def trainEM(self, n_iter, 
              out_file, 
              source_features_val=None, 
              target_features_val=None):
    for i_iter in range(n_iter):
      log_prob = self.update_counts()
      self.update_components() # XXX
      print('Iteration {}, log likelihood={}'.format(i_iter, log_prob))
      logger.info('Iteration {}, log likelihood={}'.format(i_iter, log_prob))
      if (i_iter + 1) % 5 == 0:
        with open('{}_{}_means.json'.format(out_file, i_iter), 'w') as fm,\
             open('{}_{}_transprob.json'.format(out_file, i_iter), 'w') as ft:
          json.dump(self.src_model.means.tolist(), fm, indent=4, sort_keys=True)
          json.dump(self.P_ts.tolist(), ft, indent=4, sort_keys=True)
        
        if source_features_val is not None and target_features_val is not None:
          alignments, align_probs = self.align_sents(source_features_val, target_features_val, return_align_matrix=True)        
          align_dicts = []
          for src_feat, alignment, P_a in zip(source_features_val, alignments, align_probs):
            src_sent = []
            for i in range(src_feat.shape[0]):
              log_prob_z = aligner.src_model.log_prob_z_given_X(src_feat[i])
              src_sent.append(int(np.argmax(log_prob_z)))
            align_dicts.append({'alignment': alignment.tolist(),
                                'image_concepts': src_sent,
                                'align_probs': P_a.tolist()})
          with open('{}/alignment_{}.json'.format(exp_dir, i_iter), 'w') as f:
            json.dump(align_dicts, f, indent=4, sort_keys=True)
          self.retrieve(source_features_val, target_features_val, out_file='{}_{}'.format(out_file, i_iter))

        np.save('{}_{}_means.npy'.format(out_file, i_iter), self.src_model.means)
        np.save('{}_{}_transprob.npy'.format(out_file, i_iter), self.P_ts)

  def translate_prob(self):
    return (self.alpha / self.Ks + self.trg2src_counts) / np.maximum(self.alpha + np.sum(self.trg2src_counts, axis=-1, keepdims=True), EPS)
  
  def prob_s_given_tsent(self, trg_sent):
    V_trg = to_one_hot(trg_sent, self.Kt)
    return np.mean(V_trg @ self.P_ts, axis=0) 
    
  def align_sents(self, source_feats_test,
                  target_feats_test,
                  score_type='max',
                  return_align_matrix=False): 
    alignments = []
    align_probs = []
    scores = []
    for src_feat, trg_feat in zip(source_feats_test, target_feats_test):
      trg_sent = trg_feat
      src_sent = [np.exp(self.src_model.log_prob_z_given_X(src_feat[i])) for i in range(len(src_feat))]
      V_trg = to_one_hot(trg_sent, self.Kt)
      V_src = to_one_hot(src_sent, self.Ks)
      P_a = V_trg @ self.P_ts @ V_src.T
      if score_type == 'max':
        scores.append(np.prod(np.max(P_a, axis=0)))
      elif score_type == 'mean':
        scores.append(np.prod(np.mean(P_a, axis=0)))
      else:
        raise ValueError('Score type not implemented')
      alignments.append(np.argmax(P_a, axis=0))
      if return_align_matrix:
        align_probs.append(P_a.T)

    if return_align_matrix:
      return alignments, align_probs
    return alignments, np.asarray(scores)

  def retrieve(self, source_features_test, target_features_test, out_file, kbest=10):
    n = len(source_features_test)
    print(n)
    scores = np.zeros((n, n))
    for i_utt in range(n):
      if self.use_null:
        src_feats = [source_features_test[i_utt] for _ in range(n)] 
        trg_feats = [[self.Kt - 1] + target_features_test[j_utt] for j_utt in range(n)]
      else:
        src_feats = [source_features_test[i_utt] for _ in range(n)] 
        trg_feats = [target_features_test[j_utt] for j_utt in range(n)]
       
      _, scores[i_utt] = self.align_sents(src_feats, trg_feats, score_type='max') 
    np.save('{}_scores.npy'.format(out_file), scores)
    I_kbest = np.argsort(-scores, axis=1)[:, :kbest]
    P_kbest = np.argsort(-scores, axis=0)[:kbest]
    n = len(scores)
    I_recall_at_1 = 0.
    I_recall_at_5 = 0.
    I_recall_at_10 = 0.
    P_recall_at_1 = 0.
    P_recall_at_5 = 0.
    P_recall_at_10 = 0.

    for i in range(n):
      if I_kbest[i][0] == i:
        I_recall_at_1 += 1
      
      for j in I_kbest[i][:5]:
        if i == j:
          I_recall_at_5 += 1
       
      for j in I_kbest[i][:10]:
        if i == j:
          I_recall_at_10 += 1
      
      if P_kbest[0][i] == i:
        P_recall_at_1 += 1
      
      for j in P_kbest[:5, i]:
        if i == j:
          P_recall_at_5 += 1
       
      for j in P_kbest[:10, i]:
        if i == j:
          P_recall_at_10 += 1

    I_recall_at_1 /= n
    I_recall_at_5 /= n
    I_recall_at_10 /= n
    P_recall_at_1 /= n
    P_recall_at_5 /= n
    P_recall_at_10 /= n
     
    print('Image Search Recall@1: ', I_recall_at_1)
    print('Image Search Recall@5: ', I_recall_at_5)
    print('Image Search Recall@10: ', I_recall_at_10)
    print('Captioning Recall@1: ', P_recall_at_1)
    print('Captioning Recall@5: ', P_recall_at_5)
    print('Captioning Recall@10: ', P_recall_at_10)
    logger.info('Image Search Recall@1, 5, 10: {}, {}, {}'.format(I_recall_at_1, I_recall_at_5, I_recall_at_10))
    logger.info('Captioning Recall@1, 5, 10: {}, {}, {}'.format(P_recall_at_1, P_recall_at_5, P_recall_at_10))

    fp1 = open(out_file + '_image_search.txt', 'w')
    fp2 = open(out_file + '_image_search.txt.readable', 'w')
    for i in range(n):
      I_kbest_str = ' '.join([str(idx) for idx in I_kbest[i]])
      fp1.write(I_kbest_str + '\n')
    fp1.close()
    fp2.close() 

    fp1 = open(out_file + '_captioning.txt', 'w')
    fp2 = open(out_file + '_captioning.txt.readable', 'w')
    for i in range(n):
      P_kbest_str = ' '.join([str(idx) for idx in P_kbest[:, i]])
      fp1.write(P_kbest_str + '\n\n')
      fp2.write(P_kbest_str + '\n\n')
    fp1.close()
    fp2.close()  

  def move_counts(self, k1, k2):
    self.trg2src_counts[:, k2] = self.trg2src_counts[:, k1]
    self.trg2src_counts[:, k1] = 0.

  def print_alignment(self, out_file):
    align_dicts = []
    
    for i, (src_vec_ids, trg_feat) in enumerate(zip(self.src_vec_ids_train, self.trg_feats)):
      src_feat = self.src_feats[src_vec_ids]
      alignments, P_a  = self.align_sents([src_feat], [trg_feat], return_align_matrix=True)
      alignment = alignments[0] 
      src_sent = np.argmax(self.src_model.log_prob_z(i), axis=1)
      align_dicts.append({'alignment': alignment.tolist(),
                          'image_concepts': src_sent.tolist(),
                          'align_matrix': P_a.tolist()})

    with open(out_file, 'w') as f:
      json.dump(align_dicts, f, indent=4, sort_keys=True)
  
def to_one_hot(sent, K):
  sent = np.asarray(sent)
  if len(sent.shape) < 2:
    es = np.eye(K)
    sent = np.asarray([es[int(w)] if w < K else 1./K*np.ones(K) for w in sent])
    return sent
  else:
    return sent

def load_mscoco(config, max_n_boxes=10):
  path = config
  trg_feat_file_train = path['text_caption_file_train']
  src_feat_file_train = path['image_feat_file_train']
  trg_feat_file_test = path['text_caption_file_test_retrieval']
  src_feat_file_test = path['image_feat_file_test']
  
  trg_feat_file_test_full = path['text_caption_file_test'] 
  src_feat_file_test_full = path['image_feat_file_test']
  retrieval_split = path['retrieval_split_file']
  word2idx_file = path['word_to_idx_file']
  if not os.path.isfile(trg_feat_file_test):
    with open(trg_feat_file_test_full, 'r') as f_tf,\
         open(retrieval_split, 'r') as f_r,\
         open(trg_feat_file_test, 'w') as f_tx:
      splits = f_r.read().strip().split('\n')
      trg_feat_test_full = f_tf.read().strip().split('\n')
      trg_feat_test = [line for i, line in zip(splits, trg_feat_test_full) if i == '1'] # XXX Choose the first out of five captions
      f_tx.write('\n'.join(trg_feat_test))
  
  if not os.path.isfile(word2idx_file):
    top_word_file = path['top_word_file'] 
    with open(top_word_file, 'r') as f:
      vocabs = f.read().strip().split('\n')
    
    word2idx = {w:i for i, w in enumerate(vocabs)}
    with open(word2idx_file, 'w') as f:
      json.dump(word2idx, f, indent=4, sort_keys=True)
  else:
    with open(word2idx_file, 'r') as f:
      word2idx = json.load(f) 

  with open(trg_feat_file_train, 'r') as f_tr,\
       open(trg_feat_file_test, 'r') as f_tx:
      trg_str_train = f_tr.read().strip().split('\n')
      trg_str_test = f_tx.read().strip().split('\n')
      if config.get('debug', False):
        trg_str_train = trg_str_train[:20]
        trg_str_test = trg_str_test[:20]
      trg_feats_train = [[word2idx[tw] for tw in trg_sent.split()] for trg_sent in trg_str_train] # XXX Choose the first out of five captions
      trg_feats_test = [[word2idx[tw] for tw in trg_sent.split() if tw in word2idx] for trg_sent in trg_str_test] # XXX
  
  src_feat_npz_train = np.load(src_feat_file_train)
  src_feat_npz_test = np.load(src_feat_file_test)
  print('Number of training target sentences={}, number of training source sentences={}'.format(len(trg_feats_train), len(src_feat_npz_train)))
  print('Number of test target sentences={}, number of test source sentences={}'.format(len(trg_feats_test), len(src_feat_npz_test)))

  src_feats_train = [src_feat_npz_train[k][:max_n_boxes] for k in sorted(src_feat_npz_train, key=lambda x:int(x.split('_')[-1]))] # XXX
  if len(src_feat_npz_test) > 1000:
    with open(retrieval_split, 'r') as f: 
      test_indices = [i for i, line in enumerate(f.read().strip().split('\n')) if line == '1']
      src_feats_test = [src_feat_npz_test[k][:max_n_boxes] for k in sorted(src_feat_npz_test, key=lambda x:int(x.split('_')[-1])) if int(k.split('_')[-1]) in test_indices] # XXX
  else:
    src_feats_test = [src_feat_npz_test[k][:max_n_boxes] for k in sorted(src_feat_npz_test, key=lambda x:int(x.split('_')[-1]))] # XXX

  return src_feats_train, trg_feats_train, src_feats_test, trg_feats_test

def load_flickr(path):
  trg_feat_file = path['text_caption_file']
  src_feat_file = path['image_feat_file']
  test_image_ids_file = path['test_image_ids_file']
  word_to_idx_file = path['word_to_idx_file']
  with open(word_to_idx_file, 'r') as f:
    word2idx = json.load(f)

  with open(test_image_ids_file, 'r') as f:
    test_image_ids = ['_'.join(line.split('_')[0:2]) for ex, line in enumerate(f)] # XXX

  # Load the captions
  with open(trg_feat_file, 'r') as f:
    trg_feats = []
    for ex, line in enumerate(f):
      # if ex >= 100: # XXX
      #   break
      trg_feats.append(line.split())

  # Load the image features
  src_feat_npz = np.load(src_feat_file)
  image_ids = sorted(src_feat_npz, key=lambda x:int(x.split('_')[-1])) # XXX

  # Split the features into train and test sets
  src_feats_train = []
  src_feats_test = []
  trg_feats_train = []
  trg_feats_test = []
  for ex, img_id in enumerate(image_ids):
    if img_id.split('.')[0] in test_image_ids:
      if img_id.split('_')[2] == '1': # Select one caption per image for the test set (Pick the first caption for each image)
        cur_trg_feat = [word2idx[w] for w in trg_feats[ex]] 
        trg_feats_test.append(cur_trg_feat)
        src_feats_test.append(src_feat_npz[img_id])
    else:
      cur_trg_feat = [word2idx[w] for w in trg_feats[ex]] 
      trg_feats_train.append(cur_trg_feat)
      src_feats_train.append(src_feat_npz[img_id])

  print('Number of training target sentences={}, number of training source sentences={}'.format(len(trg_feats_train), len(src_feats_train)))
  print('Number of test target sentences={}, number of test source sentences={}'.format(len(trg_feats_test), len(src_feats_test)))
  return src_feats_train, trg_feats_train, src_feats_test, trg_feats_test 
  
def load_speechcoco(path, max_n_boxes=10):
  trg_feat_file_train = path['audio_feat_file_train']
  src_feat_file_train = os.path.join(path["root"], path['image_feat_file_train'])
  trg_feat_file_test = path['audio_feat_file_test']
  src_feat_file_test = os.path.join(path["root"], path['image_feat_file_test'])
  test_image_ids_file = os.path.join(path["root"], path['retrieval_split_file'])
  codebook_file = path['audio_codebook']
  codebook = np.load(codebook_file)
  
  trg_feat_train_npz = np.load(trg_feat_file_train)
  src_feat_train_npz = np.load(src_feat_file_train)
  trg_feat_test_npz = np.load(trg_feat_file_test)
  src_feat_test_npz = np.load(src_feat_file_test)  

  with open(test_image_ids_file, 'r') as f:
    test_image_ids = [i for i, line in enumerate(f) if int(line)]

  gaussian_softmax = NegativeSquare(torch.FloatTensor(codebook), 1./30)
  trg_feats_train = []
  trg_embedding_dim = -1
  K_t = codebook.shape[0]
  for ex, k in enumerate(sorted(trg_feat_train_npz, key=lambda x:int(x.split('_')[-1]))[::2]):
    if path.get('debug', False) and len(trg_feats_train) >= 20:
      break
     
    if ex == 0:
      trg_embedding_dim = trg_feat_train_npz[k].shape[-1]
  
    if trg_feat_train_npz[k].shape[0] > 0:
      trg_feat = gaussian_softmax(torch.FloatTensor(trg_feat_train_npz[k]).unsqueeze(0), True).squeeze(0).cpu().detach().numpy()
    else:
      print('Warning: empty caption')
      trg_feat = np.zeros((1, K_t))
    trg_feats_train.append(trg_feat)

  trg_feats_test = []
  for i, k in enumerate(sorted(trg_feat_test_npz, key=lambda x:int(x.split('_')[-1]))): 
    if path.get('debug', False) and len(trg_feats_test) >= 20:
      break
    if len(trg_feat_test_npz) > 1000:
      if i in test_image_ids:
        if trg_feat_test_npz[k].shape[0] > 0:
          trg_feat = gaussian_softmax(torch.FloatTensor(trg_feat_test_npz[k]).unsqueeze(0), True).squeeze(0).cpu().detach().numpy()
        else:
          print('Warning: empty caption')
          trg_feat = np.zeros((1, K_t))
      trg_feats_test.append(trg_feat)
    else:
      if trg_feat_test_npz[k].shape[0] > 0:
          trg_feat = gaussian_softmax(torch.FloatTensor(trg_feat_test_npz[k]).unsqueeze(0), True).squeeze(0).cpu().detach().numpy()
      else:
          print('Warning: empty caption')
          trg_feat = np.zeros((1, K_t))
      trg_feats_test.append(trg_feat)
      
  train_keys = sorted(src_feat_train_npz, key=lambda x:int(x.split('_')[-1]))[::2]
  if path.get('debug', False):
    train_keys = train_keys[:20]
  src_feats_train = [src_feat_train_npz[k][:max_n_boxes] for k in train_keys]

  if len(src_feat_test_npz) > 1000:
      src_feats_test = [src_feat_test_npz[k][:max_n_boxes] for i, k in enumerate(sorted(src_feat_test_npz, key=lambda x:int(x.split('_')[-1]))) if i in test_image_ids]
  else:
      src_feats_test = [src_feat_test_npz[k][:max_n_boxes] for i, k in enumerate(sorted(src_feat_test_npz, key=lambda x:int(x.split('_')[-1])))]
  if path.get('debug', False):
    src_feats_test = src_feats_test[:20]

  print('Number of training target sentences={}, number of training source sentences={}'.format(len(trg_feats_train), len(src_feats_train)))
  print('Number of test target sentences={}, number of test source sentences={}'.format(len(trg_feats_test), len(src_feats_test)))
  return src_feats_train, trg_feats_train, src_feats_test, trg_feats_test 
   
if __name__ == '__main__':  
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('CONFIG', type=str)
  parser.add_argument('--exp_dir', '-e', type=str, default='./', help='Experimental directory')

  args = parser.parse_args()
  config = json.load(open(args.CONFIG))
  exp_dir = config.get('exp_dir', './')
  if not os.path.isdir(exp_dir):
    os.mkdir(exp_dir) 
  
  logging.basicConfig(filename='{}/train.log'.format(exp_dir), format='%(asctime)s %(message)s', level=logging.DEBUG)

  if config['dataset'] == 'mscoco':       
    src_feats_train, trg_feats_train, src_feats_test, trg_feats_test = load_mscoco(config)
    with open(config['word_to_idx_file'], 'r') as f:
      word_to_idx = json.load(f)
    Kt = len(word_to_idx)+1
    Ks = 80
    var = 160
    trg_feats_train_dict = {'arr_{}'.format(ex):np.asarray(trg_feat) for ex, trg_feat in enumerate(trg_feats_train)}
    trg_feats_test_dict = {'arr_{}'.format(ex):np.asarray(trg_feat) for ex, trg_feat in enumerate(trg_feats_test)}
  elif config['dataset'] == 'mscoco2k' or config['dataset'] == 'mscoco20k':
    Kt = Ks = 65
    var = 160
    src_feats_train, trg_feats_train, src_feats_test, trg_feats_test = load_mscoco(config)
  elif config['dataset'] == 'flickr30k':
    src_feats_train, trg_feats_train, src_feats_test, trg_feats_test = load_flickr(config)
    Kt = config.get('Kt', 2001)
    Ks = config.get('Ks', 600)
    var = config.get('var', 160)
    pretrained_vgmm_model = None
  elif config['dataset'] == 'speechcoco2k':
    Kt = Ks = config.get('Kt', 65)
    if not 'audio_codebook' in config:
      trg_feat_file_train = config['audio_feat_file_train']
      trg_feat_train_npz = np.load(trg_feat_file_train)
      X = np.concatenate([trg_feat_train_npz[k] for k in sorted(trg_feat_train_npz, key=lambda x:int(x.split('_')[-1]))], axis=0) # XXX
      codebook = KMeans(n_clusters=Kt).fit(X).cluster_centers_ 
      np.save('{}/audio_codebook.npy'.format(exp_dir), codebook)
      config['audio_codebook'] = '{}/audio_codebook.npy'.format(exp_dir)

    src_feats_train, trg_feats_train, src_feats_test, trg_feats_test = load_speechcoco(config)
    var = config.get('var', 10.)
  elif config['dataset'] == 'speechcoco':
    Kt = config.get('Kt', 400)
    Ks = config.get('Ks', 80)
    print('Start initializing audio codebook ...')
    if not 'audio_codebook' in config:
      trg_feat_file_train = config['audio_feat_file_train']
      trg_feat_train_npz = np.load(trg_feat_file_train)
      X = np.concatenate([trg_feat_train_npz[k] for k in sorted(trg_feat_train_npz, key=lambda x:int(x.split('_')[-1]))[::2]], axis=0) # XXX
      # gmm = BayesianGaussianMixture(n_components=Kt, covariance_type='diag', weight_concentration_prior=1000., max_iter=1000).fit(X)
      codebook = KMeans(n_clusters=Kt).fit(X).cluster_centers_
      np.save('{}/audio_codebook.npy'.format(exp_dir), codebook)
      config['audio_codebook'] = '{}/audio_codebook.npy'.format(exp_dir)
    print('Finish initializing the audio codebook!')

    src_feats_train, trg_feats_train, src_feats_test, trg_feats_test = load_speechcoco(config)
    var = config.get('var', 160.)

  pretrained_vgmm_model = config.get('pretrained_vgmm_model', None)
  pretrained_translateprob = config.get('pretrained_translateprob', None)
  
  aligner = FullyContinuousMixtureAligner(src_feats_train,
                                          trg_feats_train,
                                          configs={'n_trg_vocab':Kt,
                                                   'n_src_vocab':Ks,
                                                   'var':var,
                                                   'pretrained_vgmm_model':pretrained_vgmm_model,
                                                   'pretrained_translateprob':pretrained_translateprob})
  aligner.trainEM(10, '{}/mixture'.format(exp_dir), source_features_val=src_feats_test, target_features_val=trg_feats_test)
  aligner.retrieve(src_feats_test, trg_feats_test, '{}/retrieval'.format(exp_dir)) 
  alignments, align_probs = aligner.align_sents(src_feats_test, trg_feats_test, return_align_matrix=True)
  align_dicts = []
  for src_feat, alignment, P_a in zip(src_feats_test, alignments, align_probs):
    src_sent = []
    for i in range(src_feat.shape[0]):
      log_prob_z = aligner.src_model.log_prob_z_given_X(src_feat[i])
      src_sent.append(int(np.argmax(log_prob_z)))
    align_dicts.append({'alignment': alignment.tolist(),
                        'image_concepts': src_sent,
                        'align_probs': P_a.tolist()})
  with open('{}/alignment_test.json'.format(exp_dir), 'w') as f:
    json.dump(align_dicts, f, indent=4, sort_keys=True)
