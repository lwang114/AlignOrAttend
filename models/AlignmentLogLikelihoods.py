# Aurthor: Liming Wang
# Date: 2020
# Contact: limingwanggrant@gmail.com
import torch
import torch.nn as nn
from copy import deepcopy
import json
import os
import numpy as np

EPS = 1e-30
class MixtureAlignmentLogLikelihood(nn.Module):
  def __init__(self, configs):
    super(MixtureAlignmentLogLikelihood, self).__init__()
    self.Ks = configs.get('Ks', 65)
    self.Kt = configs.get('Kt', 49)
    self.L_max = configs.get('L_max', 5)
    self.T_max = configs.get('T_max', 50) 
    self.compute_softmax = configs.get('compute_softmax', True)
    self.logsoftmax = nn.LogSoftmax(dim=2)
    self.C_st = np.zeros((self.Ks, self.Kt))
    self.A = np.zeros((self.L_max, self.L_max))
    self.init =  np.zeros(self.L_max) 
    self.P_st = 1. / self.Kt * np.ones((self.Ks, self.Kt)) # Translation probability p(f_t|e_i) 

  def forward(self, src_sent, trg_sent, src_boundary, trg_boundary):
    # Inputs:
    # ------
    #   src_sent: B x L x Ks tensor [[[g_e(y) for e in range(E)] for i in range(L)] for b in range(B)], where g_e is the source encoder
    #   trg_sent: B x T x Kt tensor [[[h_f(x) for f in range(F)] for t in range(T)] for b in range(B)], where h_f is the target encoder
    #   src_boundary: B x L binary tensor storing the boundaries of words in the source sentence 
    #   trg_boundary: B x T binary tensor storing the boundaries of words in the target sentence 
    #
    # Output:
    # ------
    #   log_likelihood: float log p(x|y)
    log_likelihood = torch.zeros(1, device=src_sent.device, requires_grad=True)
    B = trg_sent.size(0) 
    
    if self.compute_softmax:
      src_sent = self.softmax(src_sent)
      trg_sent = self.softmax(trg_sent)
    
    # Compute log likelihood
    P_st = torch.FloatTensor(self.P_st).to(src_sent.device)
    prob_z_it_given_y = (src_boundary.unsqueeze(-1) * src_sent).sum(1) / src_boundary.sum(-1).unsqueeze(-1)
    # print('P_st.size, prob_z_it_given_y.size, trg_sent.size: {} {} {}'.format(P_st.size(), prob_z_it_given_y.size(), trg_sent.size()))
    prob_x_t_given_y = torch.matmul(torch.matmul(prob_z_it_given_y, P_st).unsqueeze(1), torch.transpose(trg_sent, 1, 2)).squeeze(1)
    EPStensor = torch.FloatTensor(EPS*np.ones((B, 1))).to(src_sent.device)
    log_prob_x_given_y = torch.log(torch.max(prob_x_t_given_y, EPStensor))
    log_likelihood = torch.sum(trg_boundary * log_prob_x_given_y)

    _, scales = self.compute_forward_probs(src_sent, trg_sent, src_boundary, trg_boundary)
    P_a = src_sent[0].cpu().detach().numpy() @ self.P_st @ trg_sent[0].cpu().detach().numpy().T
    log_prob = np.sum(np.log(np.mean(P_a, axis=0)))

    return log_likelihood
    
  def Estep(self, src_sent, trg_sent, src_boundary, trg_boundary, debug=False):
    forward_probs, scales = self.compute_forward_probs(src_sent, trg_sent, src_boundary, trg_boundary)  
    backward_probs = self.compute_backward_probs(src_sent, trg_sent, src_boundary, trg_boundary, scales)
    new_state_counts = forward_probs * backward_probs / np.maximum(np.sum(forward_probs * backward_probs, axis=(2, 3), keepdims=True), EPS)
    
    if self.compute_softmax:
      src_sent = src_boundary.unsqueeze(-1) * self.softmax(src_sent)
      trg_sent = trg_boundary.unsqueeze(-1) * self.softmax(trg_sent)
    src_sent = src_sent.detach().cpu().numpy()
    trg_sent = trg_sent.detach().cpu().numpy()
    Ts = torch.sum(trg_boundary, dim=-1, dtype=torch.int).detach().cpu().numpy()

    C_st = np.sum(new_state_counts, axis=2).reshape(-1, self.Ks).T @ trg_sent.reshape(-1, self.Kt)  
    self.C_st += C_st  
      
  def Mstep(self):
    self.P_st = deepcopy(self.C_st / np.maximum(np.sum(self.C_st, axis=1, keepdims=1), EPS))

  def compute_forward_probs(self, src_sent, trg_sent, src_boundary, trg_boundary):
    if len(src_sent.size()) == 2:
      src_sent = src_sent.unsqueeze(0)
      trg_sent = trg_sent.unsqueeze(0) 
      
    B = src_sent.size(0)

    if self.compute_softmax:
      src_sent = self.softmax(src_sent)
      trg_sent = self.softmax(trg_sent)
    src_sent = src_sent.detach().cpu().numpy()
    trg_sent = trg_sent.detach().cpu().numpy()

    Ts = torch.sum(trg_boundary, dim=-1, dtype=torch.int).detach().cpu().numpy()
    Ls = torch.sum(src_boundary, dim=-1, dtype=torch.int).detach().cpu().numpy()
    forward_probs = np.zeros((B, self.T_max, self.L_max, self.Ks))
    scales = np.zeros((B, self.T_max))

    for b in range(B):
      L = Ls[b]
      self.init = np.zeros(self.L_max) 
      self.A = np.zeros((self.L_max, self.L_max))
      self.init[:L] = 1. / max(L, 1) 
      self.A[:L, :L] = 1. / max(L, 1)

      T = Ts[b]
      V_src = src_sent[b]
      V_trg = trg_sent[b]
      probs_x_t_given_z = V_trg @ self.P_st.T
      forward_probs[b, 0] = np.tile(self.init[:, np.newaxis], (1, self.Ks)) * V_src * probs_x_t_given_z[0]
      scales[b, 0] = np.sum(forward_probs[b, 0])
      forward_probs[b, 0] /= np.maximum(scales[b, 0], EPS)
      for t in range(T-1):
        probs_x_t_z_given_y = V_src * probs_x_t_given_z[t+1]
        A_diag = np.diag(np.diag(self.A))
        A_offdiag = self.A - np.diag(np.diag(self.A))
        # Compute the diagonal term
        forward_probs[b, t+1] += (A_diag @ forward_probs[b, t]) * probs_x_t_given_z[t+1]
        # Compute the off-diagonal term 
        forward_probs[b, t+1] += ((A_offdiag.T @ np.sum(forward_probs[b, t], axis=-1)) * probs_x_t_z_given_y.T).T        
        scales[b, t+1] = np.sum(forward_probs[b, t+1])
        forward_probs[b, t+1] /= max(scales[b, t+1], EPS)
    return forward_probs, scales

  def compute_backward_probs(self, src_sent, trg_sent, src_boundary, trg_boundary, scales):
    if len(src_sent.size()) == 2:
      src_sent = src_sent.unsqueeze(0)
      trg_sent = trg_sent.unsqueeze(0) 
    B = src_sent.size(0)
      
    if self.compute_softmax:
      src_sent = self.softmax(src_sent)
      trg_sent = self.softmax(trg_sent)

    src_sent = src_sent.cpu().detach().numpy()
    trg_sent = trg_sent.cpu().detach().numpy()
    
    Ts = torch.sum(trg_boundary, dim=-1, dtype=torch.int).cpu().detach().numpy()
    Ls = torch.sum(src_boundary, dim=-1, dtype=torch.int).cpu().detach().numpy() 
    backward_probs = np.zeros((B, self.T_max, self.L_max, self.Ks))
    for b in range(B):
      L = Ls[b]
      self.init = np.zeros(self.L_max) 
      self.A = np.zeros((self.L_max, self.L_max))
      self.init[:L] = 1. / max(L, 1) 
      self.A[:L, :L] = 1. / max(L, 1)

      T = Ts[b]
      V_src = src_sent[b]
      V_trg = trg_sent[b] 
      probs_x_given_z = V_trg @ self.P_st.T
      backward_probs[b, T-1] = 1. / max(scales[b, T-1], EPS) 
      A_diag = np.diag(np.diag(self.A))
      A_offdiag = self.A - np.diag(np.diag(self.A))
      for t in range(T-1, 0, -1):
        prob_x_t_z_given_y = V_src * probs_x_given_z[t] 
        backward_probs[b, t-1] = A_diag @ (backward_probs[b, t] * probs_x_given_z[t]) 
        backward_probs[b, t-1] += np.tile(A_offdiag @ np.sum(backward_probs[b, t] * prob_x_t_z_given_y, axis=-1)[:, np.newaxis], (1, self.Ks))
        backward_probs[b, t-1] /= max(scales[b, t-1], EPS)
    return backward_probs  
  
  def discover(self, src_sent, trg_sent, src_boundary, trg_boundary, out_file=None):
    if len(src_sent.size()) == 2:
      src_sent = src_sent.unsqueeze(0)
      trg_sent = trg_sent.unsqueeze(0)
    # print('In discover, src_sent.size(), trg_sent.size(): {} {}'.format(src_sent.size(), trg_sent.size()))
    B = src_sent.size(0)

    if self.compute_softmax:
      src_sent = self.softmax(src_sent)
      trg_sent = self.softmax(trg_sent)

    src_sent = src_sent.cpu().detach().numpy()
    trg_sent = trg_sent.cpu().detach().numpy()

    Ls = torch.sum(src_boundary, dim=-1).cpu().detach().numpy().astype(int) 
    Ts = torch.sum(trg_boundary, dim=-1).cpu().detach().numpy().astype(int)

    # Find the optimal alignments and cluster assignment
    alignments = []
    align_probs = []
    clusters = []
    cluster_probs = []
    for b in range(B):
      L = Ls[b]
      T = Ts[b]
      V_src = src_sent[b, :L]
      V_trg = trg_sent[b, :T]
      P_a = V_src @ self.P_st @ V_trg.T / L
      alignment = np.argmax(P_a, axis=0)
      align_prob = np.prod(np.max(P_a, axis=0))
      cluster_prob = deepcopy(V_src)
      for t, i_a in enumerate(alignment):
        cluster_prob[i_a] *= self.P_st @ V_trg[t]
      cluster = np.argmax(cluster_prob, axis=1)

      alignments.append(alignment)
      clusters.append(cluster)
      align_probs.append(align_prob)
      cluster_probs.append(cluster_prob)

    if out_file:
      with open('{}_alignment.json'.format(out_file), 'w') as f:
        align_dicts = []
        for alignment, cluster in zip(alignments, clusters):
          align_dicts.append({'alignment': alignment.tolist(),
                              'cluster': cluster.tolist()})
        json.dump(align_dicts, f, indent=4, sort_keys=True)
    return alignments, clusters, align_probs, cluster_probs

  def reset(self):
    self.C_st[:] = 0. 
 
  def softmax(self, sent):
    return torch.exp(self.logsoftmax(sent))

  def save(self, out_file):
    with open('{}_translateprobs.json'.format(out_file), 'w') as f:
      json.dump(np.round(self.P_st, 2).tolist(), f, indent=4, sort_keys=True)
    np.save('{}_translateprobs.npy'.format(out_file), self.P_st)


    
class MarkovAlignmentLogLikelihood(nn.Module):
  def __init__(self, configs):
    super(MixtureAlignmentLogLikelihood, self).__init__()
    self.Ks = configs.get('Ks', 65)
    self.Kt = configs.get('Kt', 49)
    self.L_max = configs.get('L_max', 5)
    self.T_max = configs.get('T_max', 100)
    self.logsoftmax = nn.Softmax(dim=2) 
    self.P_st = 1. / self.Kt * np.ones((self.Ks, self.Kt)) # Translation probability p(f_t|e_i) 
    self.A = 1. / self.L_max * np.ones((self.L_max, self.L_max)) # Alignment transition probability p(i_t|i_t-1, L)
    self.init = 1. / self.L_max * np.ones((self.L_max)) # Alignment initial probability p(i_t|L)  
    self.C_st = np.zeros((self.Ks, self.Kt)) # TODO Counts
    self.C_trans = np.zeros((self.L_max, self.L_max))
    self.C_init = np.zeros(self.L_max)

  def forward(self, src_sent, trg_sent, src_boundary, trg_boundary):
    # Inputs:
    # ------
    #   src_sent: B x L x Ks tensor [[[g_e(y) for e in range(E)] for i in range(L)] for b in range(B)], where g_e is the source encoder
    #   trg_sent: B x T x Kt tensor [[[h_f(x) for f in range(F)] for t in range(T)] for b in range(B)], where h_f is the target encoder
    #   src_boundary: B x L binary tensor storing the boundaries of words in the source sentence 
    #   trg_boundary: B x T binary tensor storing the boundaries of words in the target sentence 
    #
    # Output:
    # ------
    #   log_likelihood: float log p(x|y)
    log_likelihood = torch.zeros(1, device=src_sent.device, requires_grad=True)

    if self.compute_softmax:
      src_sent = self.softmax(src_sent)
      trg_sent = self.softmax(trg_sent)
    trg_sent_slices = np.split(trg_sent, 1, dim=1) # Split trg_sent into T slices of [p(f_t|x_t) for b in range(B)]  

    # TODO Handle end of sentences of different lengths + handle different transition probs for different length
    forward_prob = torch.FloatTensor(self.init).to(src_sent.device).unsqueeze(-1) * src_sent
    A_diag = torch.FloatTensor(np.diag(np.diag(self.A))).to(src_sent.device)
    A_offdiag = torch.FloatTensor(A - np.diag(np.diag(self.A))).to(src_sent.device) 
    P_st = torch.FloatTensor(self.P_st).to(src_sent.device)
    for t in range(self.T_max): # Compute [p(i_t=i, z_i=k, x_1:t|y) for b in range(B)]
      prob_x_t_given_y = torch.dot(trg_sent_slices[t], P_st.T).unsqueeze(1)
      prob_x_t_z_given_y = src_sent * prob_x_t_given_y
      diag_term = torch.matmul(A_diag, forward_prob) * prob_x_t_given_y 
      offdiag_term = torch.matmul(A_offdiag.T, torch.sum(forward_prob, dim=-1)) * prob_x_t_z_given_y 
      forward_prob = diag_term + offdiag_term
    
    EPStensor = torch.FloatTensor(EPS*np.ones(B)).to(src_sent.device)
    log_likelihood = torch.sum(torch.log(torch.max(torch.sum(forward_prob, dim=(1, 2)), EPStensor))) 

    return log_likelihood
     
  def Estep(self, src_sent, trg_sent, src_boundary, trg_boundary, debug=False): # TODO Try different training schedule
    forward_probs, scales = self.compute_forward_probs(src_sent, trg_sent, src_boundary, trg_boundary)  
    backward_probs = self.compute_backward_probs(src_sent, trg_sent, src_boundary, trg_boundary, scales)
    new_state_counts = forward_probs * backward_probs / np.maximum(np.sum(forward_probs * backward_probs, axis=(2, 3), keepdims=True), EPS)
    new_trans_counts = self.compute_transition_counts(forward_probs, backward_probs, src_sent, trg_sent, src_boundary, trg_boundary)
    
    if self.compute_softmax:
      src_sent = self.softmax(src_sent)
      trg_sent = self.softmax(trg_sent)
    src_sent = src_sent.cpu().detach().numpy()
    trg_sent = trg_sent.cpu().detach().numpy()
      
    Ts = torch.sum(trg_boundary, dim=-1, dtype=torch.int).cpu().detach().numpy()
    for b, T in enumerate(Ts): # Mask variable length target sentences
      trg_sent[b, T:] = 0.

    C_st = np.sum(new_state_counts, axis=2).reshape(-1, self.Ks).T @ trg_sent.reshape(-1, self.Kt) 
    self.C_st += C_st
    self.C_init += np.sum(new_state_counts, axis=(0, 1, 3))
    self.C_trans += new_trans_counts
    
  def Mstep(self):
    self.P_st = deepcopy(self.C_st / np.maximum(np.sum(self.C_st, axis=1, keepdims=1), EPS))
    self.A = deepcopy(self.C_trans / np.maximum(np.sum(self.C_trans, axis=1, keepdims=1), EPS))
    self.init = deepcopy(self.C_init / np.maximum(np.sum(self.C_init), EPS))

  def compute_forward_probs(self, src_sent, trg_sent, src_boundary, trg_boundary):
    if len(src_sent.size()) == 2:
      src_sent = src_sent.unsqueeze(0)
      trg_sent = trg_sent.unsqueeze(0) 
    B = src_sent.size(0)

    if self.compute_softmax:
      src_sent = self.softmax(src_sent)
      trg_sent = self.softmax(trg_sent)
    src_sent = src_sent.cpu().detach().numpy()
    trg_sent = trg_sent.cpu().detach().numpy()

    Ts = torch.sum(trg_boundary, axis=-1, dtype=torch.int).cpu().detach().numpy()
    forward_probs = np.zeros((B, self.T_max, self.L_max, self.Ks))
    scales = np.zeros((B, self.T_max))

    for b in range(B):
      T = Ts[b]
      V_src = src_sent[b]
      V_trg = trg_sent[b]
      probs_x_t_given_z = V_trg @ self.P_st.T 
      forward_probs[b, 0] = np.tile(self.init[:, np.newaxis], (1, self.Ks)) * V_src * probs_x_t_given_z[0]
      scales[b, 0] = np.sum(forward_probs[b, 0])
      forward_probs[b, 0] /= max(scales[b, 0], EPS)
      for t in range(T-1):
        probs_x_t_z_given_y = V_src * probs_x_t_given_z[t+1]
        A_diag = np.diag(np.diag(self.A))
        A_offdiag = self.A - np.diag(np.diag(self.A))
        # Compute the diagonal term
        forward_probs[b, t+1] += (A_diag @ forward_probs[b, t]) * probs_x_t_given_z[t+1]
        # Compute the off-diagonal term 
        forward_probs[b, t+1] += ((A_offdiag.T @ np.sum(forward_probs[b, t], axis=-1)) * probs_x_t_z_given_y.T).T        
        scales[b, t+1] = np.sum(forward_probs[b, t+1])
        forward_probs[b, t+1] /= max(scales[b, t+1], EPS)
    return forward_probs, scales

  def compute_backward_probs(self, src_sent, trg_sent, src_boundary, trg_boundary, scales):
    if len(src_sent.size()) == 2:
      src_sent = src_sent.unsqueeze(0)
      trg_sent = trg_sent.unsqueeze(0) 
    B = src_sent.size(0)
      
    if self.compute_softmax:
      src_sent = self.softmax(src_sent)
      trg_sent = self.softmax(trg_sent)
    src_sent = src_sent.cpu().detach().numpy()
    trg_sent = trg_sent.cpu().detach().numpy()

    Ts = torch.sum(trg_boundary, dim=-1, dtype=torch.int).cpu().detach().numpy()
    backward_probs = np.zeros((B, self.T_max, self.L_max, self.Ks))
    for b in range(B):
      T = Ts[b]
      V_src = src_sent[b]
      V_trg = trg_sent[b] 
      probs_x_given_z = V_trg @ self.P_st.T
      backward_probs[b, T-1] = 1. / max(scales[b, T-1], EPS) 
      A_diag = np.diag(np.diag(self.A))
      A_offdiag = self.A - np.diag(np.diag(self.A))
      for t in range(T-1, 0, -1):
        prob_x_t_z_given_y = V_src * probs_x_given_z[t] 
        backward_probs[b, t-1] += A_diag @ (backward_probs[b, t] * probs_x_given_z[t]) 
        A_offdiag = self.A - np.diag(np.diag(self.A))
        backward_probs[b, t-1] += np.tile(A_offdiag @ np.sum(backward_probs[b, t] * prob_x_t_z_given_y, axis=-1)[:, np.newaxis], (1, self.Ks))
        backward_probs[b, t-1] /= max(scales[b, t-1], EPS)
    return backward_probs  

  def compute_transition_counts(self, forward_probs, backward_probs, src_sent, trg_sent, src_boundary, trg_boundary):
    if len(src_sent.size()) == 2:
      src_sent = src_sent.unsqueeze(0)
      trg_sent = trg_sent.unsqueeze(0) 
    B = src_sent.size(0)

    if self.compute_softmax:
      src_sent = self.softmax(src_sent)
      trg_sent = self.softmax(trg_sent)
    src_sent = src_sent.cpu().detach().numpy()
    trg_sent = trg_sent.cpu().detach().numpy()

    Ts = torch.sum(trg_boundary, dim=-1, dtype=torch.int).cpu().detach().numpy()
    transExpCounts = np.zeros((self.L_max, self.L_max))
    # Update the transition probs
    C_st = np.zeros((self.L_max, self.L_max)) 
    for b in range(B):
      T = Ts[b]
      V_src = src_sent[b]
      V_trg = trg_sent[b] 
      
      probs_x_t_given_z = V_trg @ self.P_st.T
      for t in range(T-1):
        prob_x_t_z_given_y = V_src * probs_x_t_given_z[t+1] 
        alpha = np.tile(np.sum(forward_probs[b, t], axis=-1)[:, np.newaxis], (1, self.L_max)) 
        A_diag = np.tile(np.diag(self.A)[:, np.newaxis], (1, self.nWords))
        A_offdiag = self.A - np.diag(np.diag(self.A))
        C_st += np.diag(np.sum(forward_probs[b, t] * A_diag * probs_x_t_given_z[t+1] * backward_probs[b, t+1], axis=-1))
        C_st += alpha * A_offdiag * np.sum(prob_x_t_z_given_y * backward_probs[b, t+1], axis=-1)
        
        C_st = C_st / np.sum(C_st)
        
        C_jumps = {}
        for s in range(self.L_max): # Reduce the number of parameters if the length of image-caption pairs vary too much by maintaining the Toeplitz assumption
          for next_s in range(self.L_max):
            if next_s - s not in C_jumps:
              C_jumps[next_s - s] = C_st[s, next_s]
            else:
              C_jumps[next_s - s] += C_st[s, next_s]
      
        C_st[:] = 0.
        for s in range(self.L_max):
          for next_s in range(self.L_max):
            C_st[s, next_s] += C_jumps[next_s - s]

    return C_st
   
  def softmax(self, sent):
    return torch.exp(self.logsoftmax(sent))

  def reset(self):
    self.C_st[:] = 0.
    self.C_trans[:] = 0.
    self.C_init[:] = 0.

    
def load_mscoco(path):
  trg_feat_file_train = path['text_caption_file_train']
  src_feat_file_train = path['image_feat_file_train']
  trg_feat_file_test = path['text_caption_file_test_retrieval']
  src_feat_file_test = path['image_feat_file_test_retrieval']
  
  trg_feat_file_test_full = path['text_caption_file_test'] 
  src_feat_file_test_full = path['image_feat_file_test']
  retrieval_split = path['retrieval_split_file']
  top_word_file = path['top_word_file'] 
  word2idx_file = path['word_to_idx_file']
  if not os.path.isfile(trg_feat_file_test):
    with open(trg_feat_file_test_full, 'r') as f_tf,\
         open(retrieval_split, 'r') as f_r,\
         open(trg_feat_file_test, 'w') as f_tx:
      splits = f_r.read().strip().split('\n')
      trg_feat_test_full = f_tf.read().strip().split('\n')
      trg_feat_test = [line for i, line in zip(splits, trg_feat_test_full) if i == '1']
      f_tx.write('\n'.join(trg_feat_test))
  
  if not os.path.isfile(word2idx_file):
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
      trg_feats_train = [[word2idx[tw] for tw in trg_sent.split()] for trg_sent in trg_str_train[:30]] # XXX
      trg_feats_test = [[word2idx[tw] for tw in trg_sent.split()] for trg_sent in trg_str_test[:30]] # XXX
  
  src_feat_npz_train = np.load(src_feat_file_train)
  src_feat_npz_test = np.load(src_feat_file_test)
  src_feats_train = [src_feat_npz_train[k] for k in sorted(src_feat_npz_train, key=lambda x:int(x.split('_')[-1]))[:30]] # XXX
  src_feats_test = [src_feat_npz_test[k] for k in sorted(src_feat_npz_test, key=lambda x:int(x.split('_')[-1]))[:30]] # XXX

  return src_feats_train, trg_feats_train, src_feats_test, trg_feats_test

def to_one_hot(sent, K):
  sent = np.asarray(sent)
  if len(sent.shape) < 2:
    es = np.eye(K)
    sent = np.asarray([es[int(w)] if w < K else 1./K*np.ones(K) for w in sent])
    return sent
  else:
    return sent

if __name__ == '__main__':
  from SegmentModels import NoopSegmenter
  from ImageModels import LinearEncoder
  from continuous_mixture_aligner import ContinuousMixtureAligner

  with open('../data/mscoco2k_path.json', 'r') as f:
    path = json.load(f)

  Ks = Kt = 65
  L_max = 5
  T_max = 5
  segmenter = NoopSegmenter(configs={'max_nframes': T_max,
                                     'max_nsegments': L_max})
  encoder = LinearEncoder({'n_class': Kt, 
                           'softmax_activation': 'gaussian',
                           'precision': 0.1,
                           'pretrained_model': '/ws/ifp-53_2/hasegawa/lwang114/fall2020/exp/res34_pretrained_model/image_model.14.pth',
                           'codebook_file': '/ws/ifp-53_2/hasegawa/lwang114/fall2020/exp/dnnhmmdnn_mscoco2k_word_9_27_2020/image_codebook.npy',
                           'compute_softmax': True}) 
  
  log_likelihood = MixtureAlignmentLogLikelihood(configs={'compute_softmax': False,
                                                          'Ks': Ks,
                                                          'Kt': Kt,
                                                          'T_max': T_max,
                                                          'L_max': L_max})   
  trg_sents, src_sents, _, _ = load_mscoco(path)
  src_sents = [to_one_hot(sent, 65) for sent in src_sents]  
  # src_sents = deepcopy(trg_sents)
  cont_aligner = ContinuousMixtureAligner(trg_sents, 
                                          src_sents,
                                          configs={'n_src_vocab': Kt,
                                                   'n_trg_vocab': Ks,
                                                   'pretrained_vgmm_model': '/ws/ifp-53_2/hasegawa/lwang114/fall2020/exp/dnnhmmdnn_mscoco2k_word_9_27_2020/image_codebook.npy',
                                                   'var': 10,
                                                   'use_null': False})

  src_boundary = np.zeros((len(src_sents), L_max))
  trg_boundary = np.zeros((len(trg_sents), T_max))
  src_sent_arr = np.zeros((len(src_sents), L_max, 65))
  trg_sent_arr = np.zeros((len(trg_sents), T_max, len(trg_sents[0][0])))

  for b in range(len(src_sents)):
    src_sent_arr[b, :len(src_sents[b])] = np.asarray(src_sents[b])
    trg_sent_arr[b, :len(trg_sents[b])] = np.asarray(trg_sents[b])
    src_boundary[b, :len(src_sents[b])] = 1. 
    trg_boundary[b, :len(trg_sents[b])] = 1.
    
  device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
  segmenter.to(device)
  encoder.to(device)
  log_likelihood.to(device)
  print(device)

  src_sent = torch.FloatTensor(src_sent_arr).to(device)
  trg_sent = torch.FloatTensor(trg_sent_arr).to(device)
  src_boundary = torch.FloatTensor(src_boundary).to(device)
  trg_boundary = torch.FloatTensor(trg_boundary).to(device)
  src_boundary = torch.cat((src_boundary, torch.ones((len(src_sents), 1),
                                                 device=src_sent.device)), dim=1)
  trg_boundary = torch.cat((trg_boundary, torch.ones((len(trg_sents), 1),
                                                 device=trg_sent.device)), dim=1)
  src_sent, src_mask, _ = segmenter(src_sent,
                                 src_boundary)
  trg_sent = encoder(trg_sent)
  trg_sent, trg_mask, _ = segmenter(trg_sent,
                                    trg_boundary)
  
  n_iter = 5
  for i_iter in range(n_iter):
    loss = -log_likelihood(src_sent, trg_sent, src_mask, trg_mask)
    # loss = -log_likelihood(trg_sent, src_sent, trg_mask, src_mask)
    print('Iteration {}, loss={}'.format(i_iter, loss.cpu().detach().numpy()/trg_sent.size(0)))
    log_likelihood.Estep(src_sent, trg_sent, src_mask, trg_mask, debug=True)
    log_likelihood.Mstep()

    if i_iter % 5 == 0: 
      log_likelihood.save('test_alignmentlikelihood_{}'.format(i_iter)) 
    log_likelihood.reset()

  cont_aligner.trainEM(n_iter, 'test_contaligner')
  log_likelihood.discover(src_sent, trg_sent, src_boundary, trg_boundary, 'test')

  # TODO Test gradients
