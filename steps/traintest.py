import time
import shutil
import torch
import torch.nn as nn
import numpy as np
import pickle
from .util import *
import logging
import json

logger = logging.getLogger(__name__)
def train(source_model, target_model, source_segment_model, target_segment_model, alignment_model, train_loader, test_loader, args): # XXX
    device = torch.device(args.device if torch.cuda.is_available() and args.device.split(':')[0]=='cuda' else "cpu")
    torch.set_grad_enabled(True)
    # Initialize all of the statistics we want to keep track of
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
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

    '''
    if not isinstance(target_model, torch.nn.DataParallel) and args.device == 'gpu':
         target_model = nn.DataParallel(target_model, device_ids=[device])
      
    if not isinstance(source_model, torch.nn.DataParallel) and args.device == 'gpu':
        source_model = nn.DataParallel(source_model, device_ids=[device])

    if not isinstance(alignment_model, torch.nn.DataParallel) and args.device == 'gpu':
        alignment_model = nn.DataParallel(alignment_model, device_ids=[device])
    '''
        
    if epoch != 0:
        source_model.load_state_dict(torch.load("%s/models/target_model.%d.pth" % (exp_dir, epoch)))
        target_model.load_state_dict(torch.load("%s/models/source_model.%d.pth" % (exp_dir, epoch)))
        # TODO Load parameters for the segmenter and the aligner
        print("loaded parameters from epoch %d" % epoch)

    target_model = target_model.to(device)
    source_model = source_model.to(device)
    alignment_model = alignment_model.to(device)

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

    if epoch != 0:
        optimizer.load_state_dict(torch.load("%s/models/optim_state.%d.pth" % (exp_dir, epoch)))
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        print("loaded state dict from epoch %d" % epoch)

    start_epoch = epoch + 1
    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("start training...")

    target_model.train()
    source_model.train()
    for epoch in range(start_epoch, args.n_epochs):
        adjust_learning_rate(args.lr, args.lr_decay, optimizer, epoch)
        end_time = time.time()
        target_model.train()
        source_model.train()
        
        for i, (source_input, target_input, source_segmentation, target_segmentation) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end_time)
            B = target_input.size(0)

            target_input = target_input.to(device)
            source_input = source_input.to(device)
            target_segmentation = target_segmentation.to(device)
            source_segmentation = source_segmentation.to(device)
            optimizer.zero_grad()

            # Compute source and target outputs
            if len(source_input.size()) >= 5: # Collapse the first two dimensions if image input includes multiple regions per image
                L = source_input.size(1)
                source_input = source_input.view(B*L, source_input.size(2), source_input.size(3), source_input.size(4))
                source_output = source_model(source_input)
                source_output = source_output.view(B, L, -1)
            else:
                source_output = source_model(source_input)
            
            if len(target_input.size()) >= 5:
                L = target_input.size(1)
                target_input = target_input.view(B*L, target_input.size(2), target_input.size(3), target_input.size(4))
                target_output = target_model(target_input)
                target_output = target_output.view(B, L, -1)
            else:
                target_output = target_model(target_input)
            
            source_pooling_ratio = round(source_input.size(-1) / source_output.size(-1))
            target_pooling_ratio = round(target_input.size(-1) / target_output.size(-1))
            # Downsample the source segmentation by pooling ratio
            if source_pooling_ratio > 1: 
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
            source_output, source_masks = source_segment_model.embed(source_output, source_segmentation) # TODO 
            target_output, target_masks = target_segment_model.embed(target_output, target_segmentation)
            alignment_model.EMstep(source_output, target_output, source_masks, target_masks)
            loss = -alignment_model(source_output, target_output, source_masks, target_masks)
            loss.backward()
            optimizer.step()

            # record loss
            loss_meter.update(loss.item(), B)
            batch_time.update(time.time() - end_time)

            if global_step % args.n_print_steps == 0 and global_step != 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t'
                      'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t'
                  'Loss total {loss_meter.val:.4f} ({loss_meter.avg:.4f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss_meter=loss_meter))
                logger.info('Epoch: [{0}][{1}/{2}]\t'
                            'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t'
                            'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t'
                  'Loss total {loss_meter.val:.4f} ({loss_meter.avg:.4f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss_meter=loss_meter))
                if np.isnan(loss_meter.avg):
                    print("training diverged...")
                    return
                
            end_time = time.time()
            global_step += 1

        if epoch % 10 == 0:
            # TODO Save parameters for the aligner and the segmenter
            torch.save(target_model.state_dict(),
                "%s/target_model.%d.pth" % (exp_dir, epoch))
            torch.save(source_model.state_dict(),
                "%s/source_model.%d.pth" % (exp_dir, epoch))
            torch.save(optimizer.state_dict(), "%s/optim_state.%d.pth" % (exp_dir, epoch))
            
            recalls = validate(source_model, target_model, source_segment_model, target_segment_model, alignment_model, test_loader, args)
        
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

def validate(source_model, target_model, source_segment_model, target_segment_model, alignment_model, val_loader, args):
    device = torch.device(args.device if torch.cuda.is_available() and args.device.split(':')[0]=='cuda' else "cpu")
    batch_time = AverageMeter()
    # if not isinstance(target_model, torch.nn.DataParallel):
    #     target_model = nn.DataParallel(target_model, device_ids=[device])
    # if not isinstance(source_model, torch.nn.DataParallel):
    #     source_model = nn.DataParallel(source_model, device_ids=[device])
    target_model = target_model.to(device)
    source_model = source_model.to(device)
    # switch to evaluate mode
    source_model.eval()
    target_model.eval()

    end = time.time()
    N_examples = val_loader.dataset.__len__()
    I_embeddings = [] 
    A_embeddings = [] 
    source_masks = []
    target_masks = []
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
            if len(source_input.size()) >= 5: # Collapse the first two dimensions if image input includes multiple regions per image
                L = source_input.size(1)
                source_input = source_input.view(B*L, source_input.size(2), source_input.size(3), source_input.size(4))
                source_output = source_model(source_input)
                source_output = source_output.view(B, L, -1)
            else:
                source_output = source_model(source_input)
            
            if len(target_input.size()) >= 5:
                L = target_input.size(1)
                target_input = target_input.view(B*L, target_input.size(2), source_input.size(3), source_input.size(4))
                target_output = target_model(target_input)
                target_output = target_output.view(B, L, -1)
            else:
                target_output = target_model(target_input)
            
            source_output = source_output.cpu().detach()
            target_output = target_output.cpu().detach()

            I_embeddings.append(source_output)
            A_embeddings.append(target_output)
            
            source_pooling_ratio = round(source_input.size(-1) / source_output.size(-1))
            target_pooling_ratio = round(target_input.size(-1) / target_output.size(-1))
            # Downsample the segmentations according to the pooling ratio
            if source_pooling_ratio > 1:
              source_segmentation_down = np.zeros((B, source_segmentation.size(-1) // source_pooling_ratio))
              for b in range(B):
                segments = np.nonzero(source_segmentation[b].cpu().numpy())[0] // pooling_ratio
                source_segmentation_down[b, segments] = 1
              source_segmentation = torch.FloatTensor(source_segmentation_down).to(device=device)
            
            if target_pooling_ratio > 1:
              target_segmentation_down = np.zeros((B, target_segmentation.size(-1) // target_pooling_ratio))
              for b in range(B):
                segments = np.nonzero(target_segmentation[b].cpu().numpy())[0] // pooling_ratio
                target_segmentation_down[b, segments] = 1 
              target_segmentation = torch.FloatTensor(target_segmentation_down).to(device=device) 
            
            source_output, source_mask = source_segment_model.embed(source_output, source_segmentation)
            target_output, target_mask = target_segment_model.embed(target_output, target_segmentation)

            source_masks.append(source_mask)
            target_masks.append(target_mask)
           
            alignments, clusters, _, _ = alignment_model.discover(source_output, target_output, source_mask, target_mask)
            for b, (alignment, cluster) in enumerate(zip(alignments, clusters)):
              align_results.append({'index': i*B+b,
                                    'alignment': alignment.tolist(),
                                    'image_concepts': cluster.tolist()})
            batch_time.update(time.time() - end)
            end = time.time()

        source_output = torch.cat(I_embeddings)
        target_output = torch.cat(A_embeddings)

        source_masks = torch.cat(source_masks)
        target_masks = torch.cat(target_masks)

        recalls = calc_recalls(source_output, target_output, source_masks, target_masks, alignment_model)
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
