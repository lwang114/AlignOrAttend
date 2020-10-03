import time
import shutil
import torch
import torch.nn as nn
import numpy as np
import pickle
import json
from .util import *
import pdb
import os

def train(audio_model, image_model, train_loader, test_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    if not isinstance(audio_model, torch.nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)

    if not isinstance(image_model, torch.nn.DataParallel):
        image_model = nn.DataParallel(image_model)
    
    if epoch != 0:
        audio_model.load_state_dict(torch.load("%s/models/audio_model.%d.pth" % (exp_dir, epoch)))
        image_model.load_state_dict(torch.load("%s/models/image_model.%d.pth" % (exp_dir, epoch)))
        print("loaded parameters from epoch %d" % epoch)

    audio_model = audio_model.to(device)
    image_model = image_model.to(device)
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

    epoch += 1
    
    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("start training...")

    audio_model.train()
    image_model.train()
    while epoch < args.n_epochs:
        adjust_learning_rate(args.lr, args.lr_decay, optimizer, epoch)
        end_time = time.time()
        audio_model.train()
        image_model.train()
        for i, (audio_input, image_input, nphones, nregions) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end_time)
            B = audio_input.size(0)

            audio_input = audio_input.to(device)
            image_input = image_input.to(device)
            image_input = image_input.transpose(2,1)

            optimizer.zero_grad()
            
            audio_output = audio_model(audio_input)
            image_output = image_model(image_input).unsqueeze(-1) # Make the image output 4D

            pooling_ratio = round(audio_input.size(-1) / audio_output.size(-1))
            nphones = nphones // pooling_ratio

            if args.losstype == 'triplet':
                loss = sampled_margin_rank_loss(image_output, audio_output,
                nphones, nregions=nregions, margin=args.margin, simtype=args.simtype)
            elif args.losstype == 'mml':
                loss = mask_margin_softmax_loss(image_output, audio_output,
                                                nphones, nregions=nregions, margin=args.margin, simtype=args.simtype)
            elif args.losstype == 'DAMSM':
                loss = DAMSM_loss(image_output, audio_output,
                                                nphones, nregions=nregions, margin=args.margin, simtype=args.simtype)
            loss.backward()
            optimizer.step()

            # record loss
            loss_meter.update(loss.item(), B)
            batch_time.update(time.time() - end_time)
            global_step += 1
            # if global_step % args.n_print_steps == 0 and global_step != 0: 
            if i % 500 == 0:
                info = 'Itr {} {loss_meter.val:.4f} ({loss_meter.avg:.4f} '.format(i,loss_meter=loss_meter)
                print(info)
            i += 1
            # if i==2565:
                # print(i)
                # pdb.set_trace()
        # pdb.set_trace()
        info = ('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss total {loss_meter.val:.4f} ({loss_meter.avg:.4f})').format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss_meter=loss_meter)
        print(info)

        end_time = time.time()

        if np.isnan(loss_meter.avg):
            print("training diverged...")
            return
        if epoch%5==0:
            recalls = validate(audio_model, image_model, test_loader, args)
            
            avg_acc = (recalls['A_r10'] + recalls['I_r10']) / 2

            torch.save(audio_model.state_dict(),
                    "%s/models/audio_model.%d.pth" % (exp_dir, epoch))
            torch.save(image_model.state_dict(),
                    "%s/models/image_model.%d.pth" % (exp_dir, epoch))
            torch.save(optimizer.state_dict(), "%s/models/optim_state.%d.pth" % (exp_dir, epoch))
            
            info = ' Epoch: [{0}] Loss: {loss_meter.val:.4f}  Audio: R1: {R1_:.4f} R5: {R5_:.4f}  R10: {R10_:.4f}  Image: R1: {IR1_:.4f} R5: {IR5_:.4f}  R10: {IR10_:.4f}  \n \
                    '.format(epoch,loss_meter=loss_meter,R1_=recalls['A_r1'],R5_=recalls['A_r5'],R10_=recalls['A_r10'],IR1_=recalls['I_r1'],IR5_=recalls['I_r5'],IR10_=recalls['I_r10'])
            save_path = os.path.join(exp_dir, 'result_file.txt')
            with open(save_path, "a") as file:
                file.write(info)

            if avg_acc > best_acc:
                best_epoch = epoch
                best_acc = avg_acc
                shutil.copyfile("%s/models/audio_model.%d.pth" % (exp_dir, epoch), 
                    "%s/models/best_audio_model.pth" % (exp_dir))
                shutil.copyfile("%s/models/image_model.%d.pth" % (exp_dir, epoch), 
                    "%s/models/best_image_model.pth" % (exp_dir))
            _save_progress()
        epoch += 1


def train_vector(audio_model, image_model, train_loader, test_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    
    if not isinstance(audio_model, torch.nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)

    if not isinstance(image_model, torch.nn.DataParallel):
        image_model = nn.DataParallel(image_model)
    
    if epoch != 0:
        audio_model.load_state_dict(torch.load("%s/models/audio_model.%d.pth" % (exp_dir, epoch)))
        image_model.load_state_dict(torch.load("%s/models/image_model.%d.pth" % (exp_dir, epoch)))
        print("loaded parameters from epoch %d" % epoch)

    audio_model = audio_model.to(device)
    image_model = image_model.to(device)
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

    epoch += 1
    
    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("start training...")

    audio_model.train()
    image_model.train()
    while epoch < args.n_epochs:
        adjust_learning_rate(args.lr, args.lr_decay, optimizer, epoch)
        end_time = time.time()
        audio_model.train()
        image_model.train()
        for i, (audio_input, image_input, nphones, nregions) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end_time)
            B = audio_input.size(0)

            audio_input = audio_input.to(device)
            image_input = image_input.to(device)
            image_input = image_input.mean(1)
            length = nphones.long().to(device)
            optimizer.zero_grad()

            audio_output = audio_model(audio_input,length)
            image_output = image_model(image_input) # Make the image output 4D

            # TODO
            if args.losstype == 'tripop':
                loss = sampled_margin_rank_loss_vector_opt(image_output, audio_output,
                nphones, nregions=nregions, margin=args.margin, simtype=args.simtype)
            if args.losstype == 'triplet':
                loss = sampled_margin_rank_loss_vector(image_output, audio_output,
                nphones, nregions=nregions, margin=args.margin, simtype=args.simtype)
            elif args.losstype == 'mml':
                loss = mask_margin_softmax_loss_vector(image_output, audio_output,
                                                nphones, nregions=nregions, margin=args.margin, simtype=args.simtype)
            elif args.losstype == 'DAMSM':
                loss = DAMSM_loss_vector(image_output, audio_output,
                                                nphones, nregions=nregions, margin=args.margin, simtype=args.simtype)
            loss.backward()
            optimizer.step()

            # record loss
            loss_meter.update(loss.item(), B)
            batch_time.update(time.time() - end_time)
            global_step += 1
            # if global_step % args.n_print_steps == 0 and global_step != 0: 
            if i%500 == 0:
                info = 'Itr {} {loss_meter.val:.4f} ({loss_meter.avg:.4f} '.format(i,loss_meter=loss_meter)
                print(info)
            i += 1
            # if i==2565:
                # print(i)
                # pdb.set_trace()
        # pdb.set_trace()
        info = ('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss total {loss_meter.val:.4f} ({loss_meter.avg:.4f})').format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss_meter=loss_meter)
        print(info)

        end_time = time.time()

        if np.isnan(loss_meter.avg):
            print("training diverged...")
            return
        if epoch%5==0:
            recalls = validate_vector(audio_model, image_model, test_loader, args)
            
            avg_acc = (recalls['A_r10'] + recalls['I_r10']) / 2

            torch.save(audio_model.state_dict(),
                    "%s/models/audio_model.%d.pth" % (exp_dir, epoch))
            torch.save(image_model.state_dict(),
                    "%s/models/image_model.%d.pth" % (exp_dir, epoch))
            torch.save(optimizer.state_dict(), "%s/models/optim_state.%d.pth" % (exp_dir, epoch))
            
            info = ' Epoch: [{0}] Loss: {loss_meter.val:.4f}  Audio: R1: {R1_:.4f} R5: {R5_:.4f}  R10: {R10_:.4f}  Image: R1: {IR1_:.4f} R5: {IR5_:.4f}  R10: {IR10_:.4f}  \n \
                    '.format(epoch,loss_meter=loss_meter,R1_=recalls['A_r1'],R5_=recalls['A_r5'],R10_=recalls['A_r10'],IR1_=recalls['I_r1'],IR5_=recalls['I_r5'],IR10_=recalls['I_r10'])
            save_path = os.path.join(exp_dir, 'result_file.txt')
            with open(save_path, "a") as file:
                file.write(info)

            if avg_acc > best_acc:
                best_epoch = epoch
                best_acc = avg_acc
                shutil.copyfile("%s/models/audio_model.%d.pth" % (exp_dir, epoch), 
                    "%s/models/best_audio_model.pth" % (exp_dir))
                shutil.copyfile("%s/models/image_model.%d.pth" % (exp_dir, epoch), 
                    "%s/models/best_image_model.pth" % (exp_dir))
            _save_progress()
        epoch += 1

def validate(audio_model, image_model, val_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_time = AverageMeter()
    
    if not isinstance(audio_model, torch.nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    if not isinstance(image_model, torch.nn.DataParallel):
        image_model = nn.DataParallel(image_model)
    
    audio_model = audio_model.to(device)
    image_model = image_model.to(device)
    # switch to evaluate mode
    image_model.eval()
    audio_model.eval()

    end = time.time()
    N_examples = val_loader.dataset.__len__()
    I_embeddings = [] 
    A_embeddings = [] 
    frame_counts = []
    region_counts = []
    with torch.no_grad():
        for i, (audio_input, image_input, nphones, nregions) in enumerate(val_loader):
            image_input = image_input.to(device)
            audio_input = audio_input.to(device)
            image_input = image_input.transpose(2,1)

            # compute output
            image_output = image_model(image_input).unsqueeze(-1) # Make the image output 4D
            audio_output = audio_model(audio_input)

            image_output = image_output.to('cpu').detach()
            audio_output = audio_output.to('cpu').detach()

            I_embeddings.append(image_output)
            A_embeddings.append(audio_output)
            
            pooling_ratio = round(audio_input.size(-1) / audio_output.size(-1))
            nphones = nphones // pooling_ratio

            frame_counts.append(nphones.cpu())
            region_counts.append(nregions.cpu())

            batch_time.update(time.time() - end)
            end = time.time()

        image_output = torch.cat(I_embeddings)
        audio_output = torch.cat(A_embeddings)
        nphones = torch.cat(frame_counts)
        nregions = torch.cat(region_counts)
        recalls = calc_recalls(image_output, audio_output,args, nphones, nregions=nregions, simtype=args.simtype)
        A_r10 = recalls['A_r10']
        I_r10 = recalls['I_r10']
        A_r5 = recalls['A_r5']
        I_r5 = recalls['I_r5']
        A_r1 = recalls['A_r1']
        I_r1 = recalls['I_r1']

    print(' * Audio R@10 {A_r10:.3f} Image R@10 {I_r10:.3f} over {N:d} validation pairs'
          .format(A_r10=A_r10, I_r10=I_r10, N=N_examples))
    print(' * Audio R@5 {A_r5:.3f} Image R@5 {I_r5:.3f} over {N:d} validation pairs'
          .format(A_r5=A_r5, I_r5=I_r5, N=N_examples))
    print(' * Audio R@1 {A_r1:.3f} Image R@1 {I_r1:.3f} over {N:d} validation pairs'
          .format(A_r1=A_r1, I_r1=I_r1, N=N_examples))

    return recalls

def align(audio_model, image_model, val_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_time = AverageMeter()
    
    if not isinstance(audio_model, torch.nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    if not isinstance(image_model, torch.nn.DataParallel):
        image_model = nn.DataParallel(image_model)
    
    audio_model.load_state_dict(torch.load('{}/models/best_audio_model.pth'.format(args.exp_dir)))
    image_model.load_state_dict(torch.load('{}/models/best_image_model.pth'.format(args.exp_dir)))

    audio_model = audio_model.to(device)
    image_model = image_model.to(device)
    # switch to evaluate mode
    image_model.eval()
    audio_model.eval()

    end = time.time()
    N_examples = val_loader.dataset.__len__()
    image_concept_file = args.image_concept_file
    split_file = args.datasplit
    if not split_file:
      selected_indices = list(range(N_examples))
    else:
      with open(split_file, 'r') as f:
        selected_indices = [i for i, line in enumerate(f) if int(line)]
    
    with open(image_concept_file, 'r') as f:
      image_concepts = [line.split() for line in f]

    I_embeddings = [] 
    A_embeddings = [] 
    frame_counts = []
    region_counts = []
    alignments = []
    with torch.no_grad():
        for i, (audio_input, image_input, nphones, nregions) in enumerate(val_loader):
            image_input = image_input.to(device)
            audio_input = audio_input.to(device)

            # compute output
            image_output = image_model(image_input).unsqueeze(-1) # Make the image output 4D
            audio_output = audio_model(audio_input)

            image_output = image_output.to('cpu').detach()
            audio_output = audio_output.to('cpu').detach()

            pooling_ratio = round(audio_input.size(-1) / audio_output.size(-1))
            n = image_output.size(0)
            
            for i_b in range(n):
              M = computeMatchmap(image_output[i_b], audio_output[i_b])
              alignment_out = np.argmax(M.squeeze().numpy(), axis=0).tolist()
              alignment_resampled = [i_a for i_a in alignment_out for _ in range(pooling_ratio)]
              cur_idx = selected_indices[i*n+i_b]
              alignment = alignment_resampled[:int(nphones[i_b])]
              align_info = {
                'index': cur_idx,
                'image_concepts': image_concepts[cur_idx],
                'alignment': alignment
                }
              alignments.append(align_info)
            print('Process {} batches after {}s'.format(i, time.time()-end))
    with open('{}/alignment.json'.format(args.exp_dir), 'w') as f:
      json.dump(alignments, f, indent=4, sort_keys=True)


def validate_vector(audio_model, image_model, val_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_time = AverageMeter()
    
    if not isinstance(audio_model, torch.nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    if not isinstance(image_model, torch.nn.DataParallel):
        image_model = nn.DataParallel(image_model)
    
    audio_model = audio_model.to(device)
    image_model = image_model.to(device)
    # switch to evaluate mode
    image_model.eval()
    audio_model.eval()

    end = time.time()
    N_examples = val_loader.dataset.__len__()
    I_embeddings = [] 
    A_embeddings = [] 
    frame_counts = []
    region_counts = []
    with torch.no_grad():
        for i, (audio_input, image_input, nphones, nregions) in enumerate(val_loader):
            image_input = image_input.to(device)
            audio_input = audio_input.to(device)
            length = nphones.long().to(device)
            image_input = image_input.mean(1)
            # compute output
            image_output = image_model(image_input)
            audio_output = audio_model(audio_input,length)

            image_output = image_output.to('cpu').detach()
            audio_output = audio_output.to('cpu').detach()

            I_embeddings.append(image_output)
            A_embeddings.append(audio_output)

            batch_time.update(time.time() - end)
            end = time.time()

        image_output = torch.cat(I_embeddings)
        audio_output = torch.cat(A_embeddings)
        # nphones = torch.cat(frame_counts)
        # nregions = torch.cat(region_counts)
        recalls = calc_recalls(image_output, audio_output,args,nphones)
        A_r10 = recalls['A_r10']
        I_r10 = recalls['I_r10']
        A_r5 = recalls['A_r5']
        I_r5 = recalls['I_r5']
        A_r1 = recalls['A_r1']
        I_r1 = recalls['I_r1']

    print(' * Audio R@10 {A_r10:.3f} Image R@10 {I_r10:.3f} over {N:d} validation pairs'
          .format(A_r10=A_r10, I_r10=I_r10, N=N_examples))
    print(' * Audio R@5 {A_r5:.3f} Image R@5 {I_r5:.3f} over {N:d} validation pairs'
          .format(A_r5=A_r5, I_r5=I_r5, N=N_examples))
    print(' * Audio R@1 {A_r1:.3f} Image R@1 {I_r1:.3f} over {N:d} validation pairs'
          .format(A_r1=A_r1, I_r1=I_r1, N=N_examples))

    return recalls


def evaluation(audio_model, image_model,test_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    progress_pkl = "%s/progress.pkl" % exp_dir
    progress, epoch, global_step, best_epoch, best_acc = load_progress(progress_pkl)
    print("\nResume training from:")
    print("  epoch = %s" % epoch)
    print("  global_step = %s" % global_step)
    print("  best_epoch = %s" % best_epoch)
    print("  best_acc = %.4f" % best_acc)

    
    if not isinstance(audio_model, torch.nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)

    if not isinstance(image_model, torch.nn.DataParallel):
        image_model = nn.DataParallel(image_model)
    
    
    if epoch != 0:
        audio_model.load_state_dict(torch.load("%s/models/audio_model.%d.pth" % (exp_dir, epoch)))
        image_model.load_state_dict(torch.load("%s/models/image_model.%d.pth" % (exp_dir, epoch)))
        print("loaded parameters from epoch %d" % epoch)

    audio_model = audio_model.to(device)
    image_model = image_model.to(device)
    
    recalls = validate(audio_model, image_model, test_loader, args)
    
    avg_acc = (recalls['A_r10'] + recalls['I_r10']) / 2
    
    info = ' Epoch: [{0}] Loss: {loss_meter.val:.4f}  Audio: R1: {R1_:.4f} R5: {R5_:.4f}  R10: {R10_:.4f}  Image: R1: {IR1_:.4f} R5: {IR5_:.4f}  R10: {IR10_:.4f}  \n \
            '.format(epoch,loss_meter=loss_meter,R1_=recalls['A_r1'],R5_=recalls['A_r5'],R10_=recalls['A_r10'],IR1_=recalls['I_r1'],IR5_=recalls['I_r5'],IR10_=recalls['I_r10'])
    print(info)


def evaluation_vector(audio_model, image_model, test_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    progress_pkl = "%s/progress.pkl" % exp_dir
    progress, epoch, global_step, best_epoch, best_acc = load_progress(progress_pkl)
    print("\nResume training from:")
    print("  epoch = %s" % epoch)
    print("  global_step = %s" % global_step)
    print("  best_epoch = %s" % best_epoch)
    print("  best_acc = %.4f" % best_acc)

    
    if not isinstance(audio_model, torch.nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)

    if not isinstance(image_model, torch.nn.DataParallel):
        image_model = nn.DataParallel(image_model)
    

    if epoch != 0:
        audio_model.load_state_dict(torch.load("%s/models/audio_model.%d.pth" % (exp_dir, epoch)))
        image_model.load_state_dict(torch.load("%s/models/image_model.%d.pth" % (exp_dir, epoch)))
        print("loaded parameters from epoch %d" % epoch)

    audio_model = audio_model.to(device)
    image_model = image_model.to(device)
    
    recalls = validate_vector(audio_model, image_model, test_loader, args)
    
    avg_acc = (recalls['A_r10'] + recalls['I_r10']) / 2
    
    info = ' Epoch: [{0}] Loss: {loss_meter.val:.4f}  Audio: R1: {R1_:.4f} R5: {R5_:.4f}  R10: {R10_:.4f}  Image: R1: {IR1_:.4f} R5: {IR5_:.4f}  R10: {IR10_:.4f}  \n \
            '.format(epoch,loss_meter=loss_meter,R1_=recalls['A_r1'],R5_=recalls['A_r5'],R10_=recalls['A_r10'],IR1_=recalls['I_r1'],IR5_=recalls['I_r5'],IR10_=recalls['I_r10'])
    print(info)
