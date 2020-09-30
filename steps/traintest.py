import time
import shutil
import torch
import torch.nn as nn
import numpy as np
import pickle
from .util import *
import logging
import json
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)
def train(source_model, target_model,
          source_segment_model, target_segment_model,
          alignment_model,
          train_loader, test_loader,
          args, retriever=None): 
    device = torch.device(args.device if torch.cuda.is_available() and args.device.split(':')[0]=='cuda' else "cpu") 
    torch.set_grad_enabled(True)
    # Initialize all of the statistics we want to keep track of
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    align_loss_meter = AverageMeter()
    progress = []
    best_epoch, best_acc = 0, -np.inf
    global_step, epoch = 0, 0
    start_time = time.time()
    exp_dir = args.exp_dir

    def _save_progress():
        progress.append([epoch, global_step, best_epoch, best_acc, 
                time.time() - start_time])
        with open("%s/progress.pkl" % exp_dir, "wb") as f:
            pickle.dump(progress, f)

    # create/load exp
    if args.resume:
        progress_pkl = "%s/progress.pkl" % exp_dir
        progress, epoch, global_step, best_epoch, best_acc = load_progress(progress_pkl)
        print("\nResume training from:")
        print("  epoch = %s" % epoch)
        print("  global_step = %s" % global_step)
        print("  best_epoch = %s" % best_epoch)
        print("  best_acc = %.4f" % best_acc)

    if not isinstance(target_model, torch.nn.DataParallel) and args.device == 'gpu':
         target_model = nn.DataParallel(target_model, device_ids=[device])
      
    if not isinstance(source_model, torch.nn.DataParallel) and args.device == 'gpu':
        source_model = nn.DataParallel(source_model, device_ids=[device])

    if not isinstance(alignment_model, torch.nn.DataParallel) and args.device == 'gpu':
        alignment_model = nn.DataParallel(alignment_model, device_ids=[device])
    
    if epoch != 0:
        source_model.load_state_dict(torch.load("%s/source_model.%d.pth" % (exp_dir, epoch)))
        target_model.load_state_dict(torch.load("%s/target_model.%d.pth" % (exp_dir, epoch)))
        # TODO Load parameters for the segmenter and the aligner        
        print("loaded parameters from epoch %d" % epoch)

    target_model = target_model.to(device)
    source_model = source_model.to(device)
    alignment_model = alignment_model.to(device)
    if retriever is not None:
        retriever = retriever.to(device)

    # Set up the optimizer
    target_trainables = [p for p in target_model.parameters() if p.requires_grad]
    source_trainables = [p for p in source_model.parameters() if p.requires_grad]
    trainables = target_trainables + source_trainables
    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(trainables, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(trainables, args.lr,
                                weight_decay=args.weight_decay,
                                betas=(0.95, 0.999))
    else:
        raise ValueError('Optimizer %s is not supported' % args.optim)

    start_epoch = epoch
    if epoch != 0:
        optimizer.load_state_dict(torch.load("%s/optim_state.%d.pth" % (exp_dir, epoch)))
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        print("loaded state dict from epoch %d" % epoch)

        start_epoch += 1
        print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    
    print("start training...")

    target_model.train()
    source_model.train()

    for epoch in range(start_epoch, args.n_epochs):
        adjust_learning_rate(args.lr, args.lr_decay, optimizer, epoch)
        end_time = time.time()
        target_model.train()
        source_model.train()

        # XXX
        audio_embeds = {}
        for i, (source_input, target_input, source_segmentation, target_segmentation) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end_time)
            B = target_input.size(0)
            target_input = target_input.to(device)
            source_input = source_input.to(device)
            # print(source_input.size(), target_input.size())
            target_segmentation = target_segmentation.to(device)
            source_segmentation = source_segmentation.to(device)
            optimizer.zero_grad()

            target_embed, target_output = target_model(target_input, save_features=True)
            source_embed, source_output = source_model(source_input, save_features=True)
            
            # Compute source and target outputs
            source_pooling_ratio = round(source_input.size(1) / source_output.size(1))
            target_pooling_ratio = round(target_input.size(1) / target_output.size(1))

            # Downsample the source segmentation by pooling ratio
            if source_pooling_ratio > 1: 
              source_segmentation_down = np.zeros((B, source_segmentation.size(-1) // source_pooling_ratio))
              for b in range(B):
                segments = np.nonzero(source_segmentation[b].cpu().numpy())[0] // source_pooling_ratio
                source_segmentation_down[b, segments] = 1
              source_segmentation = torch.FloatTensor(source_segmentation_down).to(device=device)

            # Downsample the target segmentation by pooling ratio
            if target_pooling_ratio > 1:
              target_segmentation_down = np.zeros((B, target_segmentation.size(-1) // target_pooling_ratio))
              for b in range(B):
                segments = np.nonzero(target_segmentation[b].cpu().numpy())[0] // target_pooling_ratio
                target_segmentation_down[b, segments] = 1 
              target_segmentation = torch.FloatTensor(target_segmentation_down).to(device=device) 

            # Convert the segmentations to masks
            source_output, source_mask, _ = source_segment_model(source_output, source_segmentation)
            source_embed, _, _ = source_segment_model(source_embed, source_segmentation, is_embed=True) # XXX
            target_output, target_mask, _ = target_segment_model(target_output, target_segmentation)
            target_embed, _, _ = target_segment_model(target_embed, target_segmentation, is_embed=True) # XXX

            align_loss = -alignment_model(source_output, target_output, source_mask, target_mask)
            if retriever is not None:
                retrieve_loss = retriever.loss(source_embed, target_embed, source_mask, target_mask) 
                loss = retrieve_loss # + align_loss # XXX
            else:
                loss = align_loss

            alignment_model.Estep(source_output, target_output, source_mask, target_mask)
                     
            loss.backward()
            optimizer.step()

            # record loss
            align_loss_meter.update(align_loss.item(), B)
            loss_meter.update(loss.item(), B)
            batch_time.update(time.time() - end_time)

            if global_step % args.n_print_steps == 0 and global_step != 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t'
                      'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t'
                      'Align loss {align_loss_meter.val:.4f} ({align_loss_meter.avg:.4f})\t'
                      'Loss total {loss_meter.val:.4f} ({loss_meter.avg:.4f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, align_loss_meter=align_loss_meter, loss_meter=loss_meter))
                logger.info('Epoch: [{0}][{1}/{2}]\t'
                            'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t'
                            'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' 
                            'Align loss {align_loss_meter.val:.4f} ({align_loss_meter.avg:.4f})\t'
                            'Loss total {loss_meter.val:.4f} ({loss_meter.avg:.4f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, align_loss_meter=align_loss_meter, loss_meter=loss_meter))
                if np.isnan(loss_meter.avg):
                    print("training diverged...")
                    return
                
            end_time = time.time()
            global_step += 1
        
        alignment_model.Mstep()
        alignment_model.reset()
        
        if (epoch + 1) % 1 == 0:
            torch.save(target_model.state_dict(),
                "%s/target_model.%d.pth" % (exp_dir, epoch))
            torch.save(source_model.state_dict(),
                "%s/source_model.%d.pth" % (exp_dir, epoch))
            torch.save(optimizer.state_dict(), "%s/optim_state.%d.pth" % (exp_dir, epoch))
            with open('{}/transprob_{}.json'.format(exp_dir, epoch), 'w') as f:
                json.dump(alignment_model.P_st.tolist(), f, indent=4, sort_keys=True)

            recalls = validate(source_model, 
                               target_model, 
                               source_segment_model, 
                               target_segment_model, 
                               alignment_model, 
                               test_loader, args,
                               retriever=retriever)
        
            avg_acc = (recalls['A_r10'] + recalls['I_r10']) / 2
            
            if avg_acc > best_acc:
                best_epoch = epoch
                best_acc = avg_acc
                shutil.copyfile("%s/target_model.%d.pth" % (exp_dir, epoch), 
                                "%s/best_target_model.pth" % (exp_dir))
                shutil.copyfile("%s/source_model.%d.pth" % (exp_dir, epoch), 
                                "%s/best_source_model.pth" % (exp_dir))
            _save_progress()
        epoch += 1

def validate(source_model, target_model, 
             source_segment_model, target_segment_model, 
             alignment_model, 
             val_loader, args,
             retriever=None):
    device = torch.device(args.device if torch.cuda.is_available() and args.device.split(':')[0]=='cuda' else "cpu")
    batch_time = AverageMeter()
    if not isinstance(target_model, torch.nn.DataParallel):
        target_model = nn.DataParallel(target_model, device_ids=[device])
    if not isinstance(source_model, torch.nn.DataParallel):
        source_model = nn.DataParallel(source_model, device_ids=[device])
    target_model = target_model.to(device)
    source_model = source_model.to(device)
    target_segment_model = target_segment_model.to(device)
    source_segment_model = source_segment_model.to(device)
    # switch to evaluate mode
    source_model.eval()
    target_model.eval()
    source_segment_model.eval()
    target_segment_model.eval()
    alignment_model.eval()
    if retriever is not None:
        retriever.eval()

    end = time.time()
    N_examples = val_loader.dataset.__len__()
    I_outputs = []
    A_outputs = []
    I_embeddings = [] 
    A_embeddings = [] 
    image_masks = []
    audio_masks = []
    frame_counts = []
    align_results = []
    with torch.no_grad():
        for i, (source_input, target_input, source_segmentation, target_segmentation) in enumerate(val_loader):
            B = source_input.size(0)
            source_input = source_input.to(device)
            target_input = target_input.to(device)
            target_segmentation = target_segmentation.to(device)
            source_segmentation = source_segmentation.to(device)

            # Compute source and target outputs
            source_embed, source_output = source_model(source_input, save_features=True)
            target_embed, target_output = target_model(target_input, save_features=True)

            source_output = source_output.cpu().detach()
            target_output = target_output.cpu().detach()

            source_pooling_ratio = round(source_input.size(1) / source_output.size(1))
            target_pooling_ratio = round(target_input.size(1) / target_output.size(1))
            # Downsample the segmentations according to the pooling ratio
            if source_pooling_ratio > 1:
              source_segmentation_down = np.zeros((B, source_segmentation.size(-1) // source_pooling_ratio))
              for b in range(B):
                segments = np.nonzero(source_segmentation[b].cpu().numpy())[0] // source_pooling_ratio
                source_segmentation_down[b, segments] = 1
              source_segmentation = torch.FloatTensor(source_segmentation_down).to(device=device)
            
            if target_pooling_ratio > 1:
              target_segmentation_down = np.zeros((B, target_segmentation.size(-1) // target_pooling_ratio))
              for b in range(B):
                segments = np.nonzero(target_segmentation[b].cpu().numpy())[0] // pooling_ratio
                target_segmentation_down[b, segments] = 1 
              target_segmentation = torch.FloatTensor(target_segmentation_down).to(device=device) 
            
            source_output, source_mask, _ = source_segment_model(source_output, source_segmentation)
            target_output, target_mask, _ = target_segment_model(target_output, target_segmentation)
            source_embed, _, _ = source_segment_model(source_embed, source_segmentation, is_embed=True)
            target_embed, _, _ = target_segment_model(target_embed, target_segmentation, is_embed=True)

            if args.translate_direction == 'sp2im':
                I_outputs.append(source_output)
                A_outputs.append(target_output)
                I_embeddings.append(source_embed)
                A_embeddings.append(target_embed)
                image_masks.append(source_mask)
                audio_masks.append(target_mask)
            else:
                A_outputs.append(source_output)
                I_outputs.append(target_output)
                A_embeddings.append(source_embed)
                I_embeddings.append(target_embed)
                audio_masks.append(source_mask)
                image_masks.append(target_mask) 
            
            alignments, clusters, _, _ = alignment_model.discover(source_output, target_output, source_mask, target_mask)
            for b, (alignment, cluster) in enumerate(zip(alignments, clusters)):
              align_results.append({'index': i*B+b,
                                    'alignment': alignment.tolist(),
                                    'image_concepts': cluster.tolist()})
            batch_time.update(time.time() - end)
            end = time.time()

        image_outputs = torch.cat(I_outputs)
        audio_outputs = torch.cat(A_outputs)
        image_embeds = torch.cat(I_embeddings)
        audio_embeds = torch.cat(A_embeddings)
        image_masks = torch.cat(image_masks)
        audio_masks = torch.cat(audio_masks)

        recalls = calc_recalls(image_outputs, audio_outputs, 
                               image_masks, audio_masks, 
                               alignment_model, args, retriever=retriever,
                               image_embeds=image_embeds, audio_embeds=audio_embeds)        

        A_r10 = recalls['A_r10']
        I_r10 = recalls['I_r10']
        A_r5 = recalls['A_r5']
        I_r5 = recalls['I_r5']
        A_r1 = recalls['A_r1']
        I_r1 = recalls['I_r1']

    print(' * Audio R@10 {A_r10:.3f} Image R@10 {I_r10:.3f} over {N:d} validation pairs'
          .format(A_r10=A_r10, I_r10=I_r10, N=N_examples))
    logger.info(' * Audio R@10 {A_r10:.3f} Image R@10 {I_r10:.3f} over {N:d} validation pairs'
          .format(A_r10=A_r10, I_r10=I_r10, N=N_examples))
    
    print(' * Audio R@5 {A_r5:.3f} Image R@5 {I_r5:.3f} over {N:d} validation pairs'
          .format(A_r5=A_r5, I_r5=I_r5, N=N_examples))
    logger.info(' * Audio R@5 {A_r5:.3f} Image R@5 {I_r5:.3f} over {N:d} validation pairs'
          .format(A_r5=A_r5, I_r5=I_r5, N=N_examples))
    
    print(' * Audio R@1 {A_r1:.3f} Image R@1 {I_r1:.3f} over {N:d} validation pairs'
          .format(A_r1=A_r1, I_r1=I_r1, N=N_examples))
    logger.info(' * Audio R@1 {A_r1:.3f} Image R@1 {I_r1:.3f} over {N:d} validation pairs'
          .format(A_r1=A_r1, I_r1=I_r1, N=N_examples))

    with open('{}/alignment.json'.format(args.exp_dir), 'w') as f:
      json.dump(align_results, f, indent=4, sort_keys=True)

    return recalls

def initialize_clusters(encode_model,
                        segment_model,
                        alignment_model,
                        train_loader,
                        model_type,
                        configs):
    out_feats = []
    for ex, batch in enumerate(train_loader):
        # if ex > 2: # XXX
        #   break
        if model_type == 'src':
            in_feat, _, in_segment, _ = batch
        elif model_type == 'trg':
            _, in_feat, _, in_segment = batch
        B = in_feat.size(0)
        out_feat = encode_model(in_feat, save_features=True)[0]

        pooling_ratio = round(in_feat.size(1) / out_feat.size(1))
        if pooling_ratio > 1: 
            out_segment = np.zeros((B, in_segment.size(-1) // pooling_ratio))
            for b in range(B):
                segment_times = np.nonzero(in_segment[b].cpu().numpy())[0] // pooling_ratio
                out_segment[b, segment_times] = 1.
            out_segment = torch.FloatTensor(out_segment).to(device=in_feat.device)
        else:
            out_segment = in_segment

        out_feat, mask, _ = segment_model(out_feat, out_segment, is_embed=True) 
        out_feat = torch.cat(torch.split(out_feat, 1), dim=1).squeeze(0)
        keep = np.nonzero(mask.cpu().numpy().flatten(order='C'))[0]
        out_feats.append(out_feat[keep].cpu().detach().numpy())
        
    out_feats = np.concatenate(out_feats)
    codebook = KMeans(n_clusters=configs['n_class']).fit(out_feats).cluster_centers_
    np.save(configs['codebook_file'], codebook)
    return nn.Parameter(torch.FloatTensor(codebook), requires_grad=False) # XXX
