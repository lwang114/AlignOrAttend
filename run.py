import argparse
import os
import time
import json
import torch
import torch.nn as nn
import numpy as np
import logging
import models
from steps.traintest import train, validate, initialize_clusters  
from dataloaders.image_audio_caption_dataset import ImageAudioCaptionDataset

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--exp_dir', '-e', type=str, help='Experimental directory')
parser.add_argument('--dataset', choices={'mscoco2k', 'mscoco20k', 'mscoco'})
parser.add_argument('--audio_model', choices={'lstm', 'tdnn', 'transformer'}, default='lstm')
parser.add_argument('--image_model', choices={'res34', 'rcnn', 'linear'}, default='res34')
parser.add_argument('--segment_level', choices={'word', 'phone'}, default='word')
parser.add_argument('--alignment_model', choices={'mixture_aligner'}, default='mixture_aligner')
parser.add_argument('--retrieval_model', choices={None, 'linear_retriever', 'dotproduct_retriever'}, default=None)
parser.add_argument('--translate_direction', choices={'sp2im', 'im2sp'}, default='sp2im', help='Translation direction (target -> source)')
parser.add_argument('--batch_size', '-b', type=int, default=16, help='Batch size')
parser.add_argument('--optim', choices={'sgd', 'adam'}, default='sgd', help='Type of optimizer')
parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
parser.add_argument('--lr-decay', type=int, default=40, help='Divide the learning rate by 10 every lr_decay epochs')
parser.add_argument('--weight-decay', type=float, default=5e-7, help='Weight decay')
parser.add_argument('--momentum', type=float, default=0.9, help='Moementum')
parser.add_argument('--n_epochs', type=int, default=20, help='Number of training epochs')
parser.add_argument('--n_print_steps', type=int, default=10, help='Print statistics every n_print_steps')
parser.add_argument('--n_word_class', type=int, default=49, help='Number of word unit classes')
parser.add_argument('--n_concept_class', type=int, default=80, help='Number of image concept classes')
parser.add_argument('--task', choices={'alignment', 'retrieval', 'both'}, default='retrieval', help='Type of tasks to evaluate')
parser.add_argument('--start_step', type=int, default=0, help='Starting step of the experiment')
parser.add_argument('--resume', action='store_true', help='Resume the experiment')
parser.add_argument('--device', choices={'cuda:0', 'cuda:1', 'cpu'}, default='cuda:0', help='Device to use')
args = parser.parse_args()

if not os.path.isdir('data'):
  os.mkdir('data')
if not os.path.isdir(args.exp_dir):
  os.mkdir(args.exp_dir)

logging.basicConfig(filename='{}/train.log'.format(args.exp_dir), format='%(asctime)s %(message)s', level=logging.DEBUG)

# Load the data paths
if not os.path.isfile('data/{}_path.json'.format(args.dataset)):
  with open('data/{}_path.json'.format(args.dataset), 'w') as f:
    root = '/ws/ifp-53_2/hasegawa/lwang114/data/mscoco/'
    kaldi_root = '/ws/ifp-53_1/hasegawa/tools/espnet/egs/discophone/ifp_lwang114/'
    if args.dataset == 'mscoco2k':
      path = {'root': root,
              'audio_root_path_train':'{}/mscoco2k/wav/'.format(root),
              'audio_root_path_test':'{}/mscoco2k/wav/'.format(root),
              'audio_root_path_train_kaldi':'{}/dump/train/deltafalse/data.json'.format(kaldi_root),
              'audio_root_path_test_kaldi':'{}/dump/dev/deltafalse/data.json'.format(kaldi_root),
              'segment_file_train':'{}/mscoco2k/{}_phone_info.json'.format(root, args.dataset),
              'segment_file_test':'{}/mscoco2k/{}_phone_info.json'.format(root, args.dataset),
              'image_root_path_train':'{}/val2014/imgs/val2014/'.format(root),
              'image_root_path_test':'{}/val2014/imgs/val2014/'.format(root),
              'bbox_file_train':'{}/mscoco2k/{}_bboxes.txt'.format(root, args.dataset),
              'bbox_file_test':'{}/mscoco2k/{}_bboxes.txt'.format(root, args.dataset)}
    elif args.dataset == 'mscoco':
        if args.image_model == 'linear':
            image_feat_file_train = '{}/train2014/mscoco_train_res34_embed512dim_with_whole_image.npz'.format(root)
            image_feat_file_test = '{}/val2014/mscoco_val_res34_embed512dim_with_whole_image.npz'.format(root)
        elif args.image_model == 'rcnn':
            image_feat_file_train = '{}/train2014/mscoco_train_rcnn_feature.npz'.format(root)
            image_feat_file_test = '{}/val2014/mscoco_val_rcnn_feature.npz'.format(root),

        path = {'root': root, 
                'audio_root_path_train':'{}/train2014/wav/'.format(root),
                'audio_root_path_test':'{}/val2014/wav/'.format(root),
                'segment_file_train':'{}/train2014/mscoco_train_word_segments.txt'.format(root, args.dataset),
                'segment_file_test':'{}/val2014/mscoco_val_word_segments.txt'.format(root, args.dataset),
                'image_root_path_train':'{}/train2014/imgs/val2014/'.format(root),
                'image_root_path_test':'{}/val2014/imgs/val2014/'.format(root),
                'bbox_file_train':image_feat_file_train,
                'bbox_file_test':image_feat_file_test,
                'retrieval_split_file':'{}/val2014/mscoco_val_split.txt'.format(root)}
    else:
      raise ValueError('Dataset {} not supported yet'.format(args.dataset))
    json.dump(path, f, indent=4, sort_keys=True)
else:
  with open('data/{}_path.json'.format(args.dataset), 'r') as f:
    path = json.load(f) 

if args.dataset == 'mscoco2k' or args.dataset == 'mscoco20k':
  if args.segment_level == 'phone':
    args.n_word_class = 48
    args.n_concept_class = 65
  elif args.segment_level == 'word':
    args.n_concept_class = args.n_word_class = 65

if os.path.isfile('{}/model_configs.json'.format(args.exp_dir)): 
  with open('{}/model_configs.json'.format(args.exp_dir), 'r') as f:
    model_configs = json.load(f)
    configs = model_configs['batch']
    image_encoder_configs = model_configs['image_encoder']
    audio_encoder_configs = model_configs['audio_encoder']
    image_segmenter_configs = model_configs['image_segmenter']
    audio_segmenter_configs = model_configs['audio_segmenter']
    aligner_configs = model_configs['aligner']  
    retriever_configs = model_configs.get('retriever', {})
else:
  configs = {'max_num_regions': 10,
             'max_num_phones': 50,
             'max_num_frames': 800,
             'segment_level': args.segment_level, 
             'n_mfcc': 83 if args.audio_model == 'transformer' else 40}
  
  if args.translate_direction == 'sp2im':
    configs['image_first'] = True
    aligner_configs = {'Ks': args.n_concept_class,
                       'Kt': args.n_word_class,
                       'L_max': configs['max_num_regions'],
                       'T_max': configs['max_num_phones']}
  elif args.translate_direction == 'im2sp':
    configs['image_first'] = False
    aligner_configs = {'Ks': args.n_word_class,
                       'Kt': args.n_concept_class,
                       'L_max': configs['max_num_phones'],
                       'T_max': configs['max_num_regions']}

  image_encoder_configs = {'n_class': args.n_concept_class,
                           'embedding_dim': 512 if args.image_model == 'res34' or args.image_model == 'linear' else 2048,
                           'softmax_activation': 'gaussian',
                           'precision': 0.1,
                           'pretrained_model': '/ws/ifp-53_2/hasegawa/lwang114/fall2020/exp/res34_pretrained_model/image_model.14.pth',
                           'codebook_file': '{}/image_codebook.npy'.format(args.exp_dir)}

  if args.audio_model == 'tdnn':
      acoustic_embedding_dim = 512
  elif args.audio_model == 'lstm':
      acoustic_embedding_dim = 100
  elif args.audio_model == 'transformer':
      acoustic_embedding_dim = 256

  audio_encoder_configs = {'n_class': args.n_word_class,
                           'softmax_activation': 'gaussian',
                           'precision': 0.1,
                           'input_dim': configs['n_mfcc'],   
                           'embedding_dim': acoustic_embedding_dim,
                           'return_empty': False if args.segment_level == 'phone' else True,
                           'pretrained_model': '/ws/ifp-53_2/hasegawa/lwang114/summer2020/exp/blstm3_mscoco_train_sgd_lr_0.00001_mar25/audio_model.7.pth',
                           'codebook_file': '{}/audio_codebook.npy'.format(args.exp_dir)}
  audio_segmenter_configs = {'max_nframes': configs['max_num_frames'],
                             'max_nsegments': configs['max_num_phones']}
  image_segmenter_configs = {'max_nframes': configs['max_num_regions'],
                             'max_nsegments': configs['max_num_regions']}
  retriever_configs = {}
  if args.retrieval_model is not None:
    if args.translate_direction == 'sp2im':
        retriever_configs = {'embedding_dim': 2*audio_encoder_configs['embedding_dim'] if args.audio_model == 'lstm' else audio_encoder_configs['embedding_dim'],
                'input_dim': image_encoder_configs['embedding_dim']}  
    else:
        retriever_configs = {'input_dim': 2*audio_encoder_configs['embedding_dim'] if args.audio_model == 'lstm' else audio_encoder_configs['embedding_dim'],
                'embedding_dim': image_encoder_configs['embedding_dim']}  

  model_configs = {'batch': configs,
                   'image_encoder': image_encoder_configs,
                   'audio_encoder': audio_encoder_configs,
                   'image_segmenter': image_segmenter_configs,
                   'audio_segmenter': audio_segmenter_configs,
                   'aligner': aligner_configs,
                   'retriever': retriever_configs}
  with open('{}/model_configs.json'.format(args.exp_dir), 'w') as f:
    json.dump(model_configs, f, indent=4, sort_keys=True)

print(model_configs)

# Set up the dataloaders
if args.audio_model == 'transformer':
  train_loader = torch.utils.data.DataLoader(
    ImageAudioCaptionDataset(path['audio_root_path_train_kaldi'], path['image_root_path_train'], path['segment_file_train'], path['bbox_file_train'], configs=configs),
    batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True
  ) # XXX
  test_loader = torch.utils.data.DataLoader(
    ImageAudioCaptionDataset(path['audio_root_path_test_kaldi'], path['image_root_path_test'], path['segment_file_test'], path['bbox_file_test'], configs=configs),
    batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True
  )
else:
  train_loader = torch.utils.data.DataLoader(
    ImageAudioCaptionDataset(path['audio_root_path_train'], path['image_root_path_train'], path['segment_file_train'], path['bbox_file_train'], configs=configs),
    batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True
  ) # XXX
  test_loader = torch.utils.data.DataLoader(  
    ImageAudioCaptionDataset(path['audio_root_path_test'], path['image_root_path_test'], path['segment_file_test'], path['bbox_file_test'], keep_index_file=path['retrieval_split_file'], configs=configs),
    batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True
  )

# Initialize the image and audio encoders
if args.audio_model == 'tdnn':
  audio_model = models.DavenetEncoder(audio_encoder_configs)
elif args.audio_model == 'lstm':
  audio_model = models.BLSTMEncoder(audio_encoder_configs)
elif args.audio_model == 'transformer': # TODO
  pretrained_model_file = '/ws/ifp-53_1/hasegawa/tools/espnet/egs/discophone/ifp_lwang114/dump/mscoco/eval/deltafalse/split1utt/data_encoder.pth'
  audio_model = models.Transformer(n_class=49,
                            pretrained_model_file=pretrained_model_file)
  
if args.image_model == 'res34':
  image_model = models.ResnetEncoder(image_encoder_configs)
elif args.image_model == 'linear' or args.image_model == 'rcnn':
  image_model = models.LinearEncoder(image_encoder_configs)

audio_segment_model = models.NoopSegmenter(audio_segmenter_configs)
image_segment_model = models.NoopSegmenter(image_segmenter_configs)
                                    
if args.alignment_model == 'mixture_aligner':
  if args.translate_direction == 'sp2im':
    alignment_model = models.MixtureAlignmentLogLikelihood(aligner_configs)
  elif args.translate_direction == 'im2sp':
    alignment_model = models.MixtureAlignmentLogLikelihood(aligner_configs)

retriever = None
if args.retrieval_model is not None:
    if args.retrieval_model == 'linear_retriever':
        retriever = models.LinearRetriever(retriever_configs)
    elif args.retrieval_model == 'dotproduct_retriever':
        retriever = models.DotProductRetriever(retriever_configs)

if args.start_step <= 0:
  # Initialize acoustic and visual codebooks. This step may take a while, therefore save the codebooks to skip this step the next time
  print('Initializing acoustic and visual codebooks ...')
  begin_time = time.time()

  if os.path.isfile(audio_encoder_configs['codebook_file']):
    codebook = np.load(audio_encoder_configs['codebook_file'])
    audio_model.codebook = nn.Parameter(torch.FloatTensor(codebook), 
                                        requires_grad = False) # XXX
  else:
    audio_model.codebook = initialize_clusters(audio_model,
                                               audio_segment_model,
                                               alignment_model,
                                               train_loader,
                                               model_type='trg' if args.translate_direction=='sp2im' else 'src',
                                               configs=audio_encoder_configs)
  print('Finish initializing acoustic codebook after {:5f} s'.format(time.time() - begin_time))

  if os.path.isfile(image_encoder_configs['codebook_file']):
    codebook = np.load(image_encoder_configs['codebook_file']) 
    image_model.codebook = nn.Parameter(torch.FloatTensor(codebook),
                                        requires_grad = False) # XXX
  else:
    image_model.codebook = initialize_clusters(image_model,
                                             image_segment_model,
                                             alignment_model,
                                             train_loader,
                                             model_type='src' if args.translate_direction=='sp2im' else 'trg',
                                             configs=image_encoder_configs)
  print('Finish initializing visual codebook after {:5f} s'.format(time.time() - begin_time))
  
if args.start_step <= 1:
  # Train the model
  if args.translate_direction == 'sp2im':
    train(image_model,
          audio_model,
          image_segment_model,
          audio_segment_model,
          alignment_model,
          train_loader,
          test_loader,
          args,
          retriever)
  elif args.translate_direction == 'im2sp':
    train(audio_model,
          image_model,
          audio_segment_model,
          image_segment_model,
          alignment_model,
          train_loader,
          test_loader,
          args,
          retriever)

if args.start_step <= 2:
  # TODO
  # Evaluate the model
  print('to-do: evaluate word discovery performance')
