import math
import pickle
import numpy as np
import torch
import torch.nn as nn

def calc_recalls(image_outputs, 
                 audio_outputs, 
                 image_masks, 
                 audio_masks, 
                 alignment_model,
                 args,
                 retriever=None,
                 image_embeds=None,
                 audio_embeds=None):
    """
	Computes recall at 1, 5, and 10 given encoded image and audio outputs.
	"""
    n = image_outputs.size(0)
    S = np.zeros((n, n)) 
    for i in range(n):
        cur_audio_outputs = torch.cat([audio_outputs[i].unsqueeze(0) for j in range(n)])
        cur_audio_masks = torch.cat([audio_masks[i].unsqueeze(0) for j in range(n)])
        if args.task == 'alignment' or args.task == 'both':
            if args.translate_direction == 'sp2im':
                _, _, S[i], _ = alignment_model.discover(image_outputs, cur_audio_outputs, image_masks, cur_audio_masks)
            else:
                _, _, S[i], _ = alignment_model.discover(cur_audio_outputs, image_outputs, cur_audio_masks, image_masks)
        
    S = torch.FloatTensor(S)
    if retriever is not None and image_embeds is not None and audio_embeds is not None:
        S_r = retriever(image_embeds, audio_embeds, image_masks, audio_masks)
        S_A2I = S_r # XXX S * S_r.softmax(0)
        S_I2A = S_r # XXX S * S_r.softmax(1)
    else:
        S_A2I = S_I2A = S
    np.save('{}/similarity_matrix.npy'.format(args.exp_dir), S_A2I.cpu().detach().numpy())

    A2I_scores, A2I_ind = S_A2I.topk(10, 0)
    I2A_scores, I2A_ind = S_I2A.topk(10, 1)
    A_r1 = AverageMeter()
    A_r5 = AverageMeter()
    A_r10 = AverageMeter()
    I_r1 = AverageMeter()
    I_r5 = AverageMeter()
    I_r10 = AverageMeter()
    for i in range(n):
        A_foundind = -1
        I_foundind = -1
        for ind in range(10):
            if A2I_ind[ind, i] == i:
                I_foundind = ind
            if I2A_ind[i, ind] == i:
                A_foundind = ind
        # do r1s
        if A_foundind == 0:
            A_r1.update(1)
        else:
            A_r1.update(0)
        if I_foundind == 0:
            I_r1.update(1)
        else:
            I_r1.update(0)
        # do r5s
        if A_foundind >= 0 and A_foundind < 5:
            A_r5.update(1)
        else:
            A_r5.update(0)
        if I_foundind >= 0 and I_foundind < 5:
            I_r5.update(1)
        else:
            I_r5.update(0)
        # do r10s
        if A_foundind >= 0 and A_foundind < 10:
            A_r10.update(1)
        else:
            A_r10.update(0)
        if I_foundind >= 0 and I_foundind < 10:
            I_r10.update(1)
        else:
            I_r10.update(0)

    recalls = {'A_r1':A_r1.avg, 'A_r5':A_r5.avg, 'A_r10':A_r10.avg,
                'I_r1':I_r1.avg, 'I_r5':I_r5.avg, 'I_r10':I_r10.avg}
                #'A_meanR':A_meanR.avg, 'I_meanR':I_meanR.avg}

    return recalls

def computeMatchmap(I, A):
    assert(I.dim() == 3)
    assert(A.dim() == 2)
    D = I.size(0)
    H = I.size(1)
    W = I.size(2)
    T = A.size(1)                                                                                                                     
    Ir = I.view(D, -1).t()
    matchmap = torch.mm(Ir, A)
    matchmap = matchmap.view(H, W, T)  
    return matchmap

def matchmapSim(M, simtype):
    assert(M.dim() == 3)
    if simtype == 'SISA':
        return M.mean()
    elif simtype == 'MISA':
        M_maxH, _ = M.max(0)
        M_maxHW, _ = M_maxH.max(0)
        return M_maxHW.mean()
    elif simtype == 'SIMA':
        M_maxT, _ = M.max(2)
        return M_maxT.mean()
    else:
        raise ValueError

def sampled_margin_rank_loss(image_outputs, audio_outputs, nframes, margin=1., simtype='MISA', nregions=None):
    """
    image_outputs: B x D x H x W tensor
    audio_outputs: B x D x T  tensor
    Computes the triplet margin ranking loss for each anchor image/caption pair
    The impostor image/caption is randomly sampled from the minibatch
    """
    assert(image_outputs.dim() == 4)
    assert(audio_outputs.dim() == 3)
    n = image_outputs.size(0)
    loss = torch.zeros(1, device=image_outputs.device, requires_grad=True)
    for i in range(n):
        I_imp_ind = i
        A_imp_ind = i
        while I_imp_ind == i:
            I_imp_ind = np.random.randint(0, n)
        while A_imp_ind == i:
            A_imp_ind = np.random.randint(0, n)
        nF = nframes[i]
        nFimp = nframes[A_imp_ind]
        if len(nregions):
          nR = nregions[i]
          nRimp = nregions[I_imp_ind]
          anchorsim = matchmapSim(computeMatchmap(image_outputs[i][:, 0:nR], audio_outputs[i][:, 0:nF]), simtype)
          Iimpsim = matchmapSim(computeMatchmap(image_outputs[I_imp_ind][:, 0:nRimp], audio_outputs[i][:, 0:nF]), simtype)
          Aimpsim = matchmapSim(computeMatchmap(image_outputs[i][:, 0:nR], audio_outputs[A_imp_ind][:, 0:nFimp]), simtype)
        else:
          anchorsim = matchmapSim(computeMatchmap(image_outputs[i], audio_outputs[i][:, 0:nF]), simtype)
          Iimpsim = matchmapSim(computeMatchmap(image_outputs[I_imp_ind], audio_outputs[i][:, 0:nF]), simtype)
          Aimpsim = matchmapSim(computeMatchmap(image_outputs[i], audio_outputs[A_imp_ind][:, 0:nFimp]), simtype)
        A2I_simdif = margin + Iimpsim - anchorsim
        if (A2I_simdif.data > 0).all():
            loss = loss + A2I_simdif
        I2A_simdif = margin + Aimpsim - anchorsim
        if (I2A_simdif.data > 0).all():
            loss = loss + I2A_simdif
    loss = loss / n
    return loss

def mask_margin_softmax_loss(image_outputs, audio_outputs, nframes, margin=0.001, simtype='MISA', nregions=None):
    """
    image_outputs: B x D x H x W tensor
    audio_outputs: B x D x T  tensor
    Computes the masked margin softmax loss for each anchor image/caption pair as in:
    G. Ilharco, Y. Zhang, J. Baldridge. ``Large-scale representation learning from visually grounded untranscribed speech``. CoNLL, 2019.
    """
    # loss = torch.zeros(1, device=image_outputs.device, requires_grad=True)
    S = compute_matchmap_similarity_matrix(image_outputs, audio_outputs, nframes, simtype=simtype, nregions=nregions)
    m = nn.LogSoftmax() 
    n = image_outputs.size(0)
    loss = -torch.sum(m(S).diag())-torch.sum(m(S.T).diag())
    loss = loss / n
    return loss

def compute_matchmap_similarity_matrix(image_outputs, audio_outputs, nframes, simtype='MISA', nregions=None):
    """
    Assumes image_outputs is a (batchsize, embedding_dim, rows, height) tensor
    Assumes audio_outputs is a (batchsize, embedding_dim, 1, time) tensor
    Returns similarity matrix S where images are rows and audios are along the columns
    """
    assert(image_outputs.dim() == 4)
    assert(audio_outputs.dim() == 3)
    n = image_outputs.size(0)
    S = torch.zeros(n, n, device=image_outputs.device)
    for image_idx in range(n):
            for audio_idx in range(n):
                nF = max(1, nframes[audio_idx])
                if len(nregions):
                  nR = max(1, nregions[image_idx])
                  S[image_idx, audio_idx] = matchmapSim(computeMatchmap(image_outputs[image_idx][:, 0:nR], audio_outputs[audio_idx][:, 0:nF]), simtype)
                else:
                  S[image_idx, audio_idx] = matchmapSim(computeMatchmap(image_outputs[image_idx][:, 0:nR], audio_outputs[audio_idx][:, 0:nF]), simtype)
 
    return S

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(base_lr, lr_decay, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every lr_decay epochs"""
    lr = base_lr * (0.1 ** (epoch // lr_decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def load_progress(prog_pkl, quiet=False):
    """
    load progress pkl file
    Args:
        prog_pkl(str): path to progress pkl file
    Return:
        progress(list):
        epoch(int):
        global_step(int):
        best_epoch(int):
        best_avg_r10(float):
    """
    def _print(msg):
        if not quiet:
            print(msg)

    with open(prog_pkl, "rb") as f:
        prog = pickle.load(f)
        epoch, global_step, best_epoch, best_avg_r10, _ = prog[-1]

    _print("\nPrevious Progress:")
    msg =  "[%5s %7s %5s %7s %6s]" % ("epoch", "step", "best_epoch", "best_avg_r10", "time")
    _print(msg)
    return prog, epoch, global_step, best_epoch, best_avg_r10
