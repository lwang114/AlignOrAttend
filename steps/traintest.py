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
def train(audio_model, image_model, alignment_model, train_loader, test_loader, args):
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

    # if not isinstance(audio_model, torch.nn.DataParallel) and args.device == 'gpu':
    #     audio_model = nn.DataParallel(audio_model, device_ids=[device]) # XXX
            
    # if not isinstance(image_model, torch.nn.DataParallel) and args.device == 'gpu':
    #     image_model = nn.DataParallel(image_model, device_ids=[device]) # XXX

    # if not isinstance(alignment_model, torch.nn.DataParallel) and args.device == 'gpu':
    #     alignment_model = nn.DataParallel(alignment_model, device_ids=[device]) # XXX
        
    if epoch != 0:
        audio_model.load_state_dict(torch.load("%s/models/audio_model.%d.pth" % (exp_dir, epoch)))
        image_model.load_state_dict(torch.load("%s/models/image_model.%d.pth" % (exp_dir, epoch)))
        print("loaded parameters from epoch %d" % epoch)

    audio_model = audio_model.to(device)
    image_model = image_model.to(device)
    alignment_model = alignment_model.to(device)

    # Set up the optimizer
    audio_trainables = [p for p in audio_model.parameters() if p.requires_grad]
    image_trainables = [p for p in image_model.parameters() if p.requires_grad]
    trainables = audio_trainables + image_trainables
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

    audio_model.train()
    image_model.train()
    for epoch in range(start_epoch, args.n_epochs):
        adjust_learning_rate(args.lr, args.lr_decay, optimizer, epoch)
        end_time = time.time()
        audio_model.train()
        image_model.train()
        # TODO Make the definition of region masks more consistent with the original definition 
        for i, (image_input, audio_input, region_mask, phone_boundary) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end_time)
            B = audio_input.size(0)

            audio_input = audio_input.to(device)
            image_input = image_input.to(device)
            phone_boundary = phone_boundary.to(device)
            region_mask = region_mask.to(device)
            optimizer.zero_grad()

            audio_output = audio_model(audio_input)
            if len(image_input.size()) >= 5: # Collapse the first two dimensions if image input includes multiple regions per image
                L = image_input.size(1)
                image_input = image_input.view(B*L, image_input.size(2), image_input.size(3), image_input.size(4))
                image_output = image_model(image_input)
                image_output = image_output.view(B, L, -1)
            else:
                image_output = image_model(image_input)

            pooling_ratio = round(audio_input.size(-1) / audio_output.size(-1))
            if pooling_ratio > 1:
              phone_boundary_down = np.zeros((B, phone_boundary.size(-1) // pooling_ratio))
              for b in range(B):
                segments = np.nonzero(phone_boundary[b].cpu().numpy())[0] // pooling_ratio
                phone_boundary_down[b, segments] = 1 
              phone_boundary = torch.FloatTensor(phone_boundary_down).to(device=device) 

            alignment_model.EMstep(image_output, audio_output, region_mask, phone_boundary)
            loss = -alignment_model(image_output, audio_output, region_mask, phone_boundary)
            # loss.backward()
            # optimizer.step()

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
            torch.save(audio_model.state_dict(),
                "%s/audio_model.%d.pth" % (exp_dir, epoch))
            torch.save(image_model.state_dict(),
                "%s/image_model.%d.pth" % (exp_dir, epoch))
            torch.save(optimizer.state_dict(), "%s/optim_state.%d.pth" % (exp_dir, epoch))
            recalls = validate(audio_model, image_model, alignment_model, test_loader, args)
        
            avg_acc = (recalls['A_r10'] + recalls['I_r10']) / 2
        
            if avg_acc > best_acc:
                best_epoch = epoch
                best_acc = avg_acc
                shutil.copyfile("%s/audio_model.%d.pth" % (exp_dir, epoch), 
                                "%s/best_audio_model.pth" % (exp_dir))
                shutil.copyfile("%s/image_model.%d.pth" % (exp_dir, epoch), 
                                "%s/best_image_model.pth" % (exp_dir))
        _save_progress()
        epoch += 1

def validate(audio_model, image_model, alignment_model, val_loader, args):
    device = torch.device(args.device if torch.cuda.is_available() and args.device.split(':')[0]=='cuda' else "cpu")
    batch_time = AverageMeter()
    # if not isinstance(audio_model, torch.nn.DataParallel):
    #     audio_model = nn.DataParallel(audio_model, device_ids=[device])
    # if not isinstance(image_model, torch.nn.DataParallel):
    #     image_model = nn.DataParallel(image_model, device_ids=[device])
    audio_model = audio_model.to(device)
    image_model = image_model.to(device)
    # switch to evaluate mode
    image_model.eval()
    audio_model.eval()

    end = time.time()
    N_examples = val_loader.dataset.__len__()
    I_embeddings = [] 
    A_embeddings = [] 
    region_masks = []
    phone_boundaries = []
    frame_counts = []
    align_results = []
    with torch.no_grad():
        for i, (image_input, audio_input, region_mask, phone_boundary) in enumerate(val_loader):
            B = image_input.size(0)
            image_input = image_input.to(device)
            audio_input = audio_input.to(device)
            phone_boundary = phone_boundary.to(device)
            region_mask = region_mask.to(device)

            # compute output
            if len(image_input.size()) >= 5: # Collapse the first two dimensions if image input includes multiple regions per image
                L = image_input.size(1)
                image_input = image_input.view(B*L, image_input.size(2), image_input.size(3), image_input.size(4))
                image_output = image_model(image_input)
                image_output = image_output.view(B, L, -1)
            else:
                image_output = image_model(image_input)
            audio_output = audio_model(audio_input)

            image_output = image_output.cpu().detach()
            audio_output = audio_output.cpu().detach()

            I_embeddings.append(image_output)
            A_embeddings.append(audio_output)
            region_masks.append(region_mask)
            phone_boundaries.append(phone_boundary)
            
            pooling_ratio = round(audio_input.size(-1) / audio_output.size(-1))
            # Downsample the phone boundary according to the pooling ratio
            if pooling_ratio > 1:
              phone_boundary_down = np.zeros((B, phone_boundary.size(-1) // pooling_ratio))
              for b in range(B):
                segments = np.nonzero(phone_boundary[b].cpu().numpy())[0] // pooling_ratio
                phone_boundary_down[b, segments] = 1 
              phone_boundary = torch.tensor(phone_boundary_down, device=device) 
            
            alignments, clusters, _, _ = alignment_model.discover(image_output, audio_output, region_mask, phone_boundary)
            for b, (alignment, cluster) in enumerate(zip(alignments, clusters)):
              align_results.append({'index': i*B+b,
                                    'alignment': alignment.tolist(),
                                    'image_concepts': cluster.tolist()})
            batch_time.update(time.time() - end)
            end = time.time()

        image_output = torch.cat(I_embeddings)
        audio_output = torch.cat(A_embeddings)
        region_masks = torch.cat(region_masks)
        phone_boundaries = torch.cat(phone_boundaries)
        recalls = calc_recalls(image_output, audio_output, region_masks, phone_boundaries, alignment_model)
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
