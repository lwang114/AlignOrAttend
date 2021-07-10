# Modified from https://github.com/dharwath/DAVEnet-pytorch.git
import argparse
import os
import pickle
import sys
import time
import torch
import dataloaders
import models
from steps.traintest_attention import train_attention, validate_attention, evaluation_attention, align_attention
from steps.traintest_phone import train, validate, align, train_vector, evaluation,evaluation_vector
import numpy as np
import json
import random

random.seed(2)
np.random.seed(2)

print("I am process %s, running on %s: starting (%s)" % (
        os.getpid(), os.name, time.asctime()))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data_dir",type=str, default="/ws/ifp-53_2/hasegawa/lwang114/data/mscoco/")
parser.add_argument("--exp-dir", type=str, default="outputs",
        help="directory to dump experiments")
parser.add_argument("--resume",type=bool,default=False)
parser.add_argument("--dataset", type=str, choices={'flickr', 'mscoco'}, default='mscoco')
parser.add_argument("--optim", type=str, default="sgd",
        help="training optimizer", choices=["sgd", "adam"])
parser.add_argument('-b', '--batch-size', default=32, type=int,
    metavar='N', help='mini-batch size (default: 32)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
    metavar='LR', help='initial learning rate')
parser.add_argument('--lr-decay', default=40, type=int, metavar='LRDECAY',
    help='Divide the learning rate by 10 every lr_decay epochs')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-7, type=float,
    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument("--n_epochs", type=int, default=100,
        help="number of maximum training epochs")
parser.add_argument("--n_print_steps", type=int, default=10,
        help="number of steps to print statistics")
parser.add_argument("--audio-model", type=str, default="Davenet", choices=['Davenet', 'RNN'],
        help="audio model architecture")
parser.add_argument("--image-model", type=str, default="VGG16",
        help="image model architecture", choices=["VGG16"])
parser.add_argument("--pretrained-image-model", action="store_true",
    dest="pretrained_image_model", help="Use an image network pretrained on ImageNet")
parser.add_argument("--margin", type=float, default=1.0, help="Margin paramater for triplet loss")
parser.add_argument("--simtype", type=str, default="ASISA",
        help="matchmap similarity function", choices=["ASISA", "BASISA"])
parser.add_argument('--losstype', choices=['triplet', 'mml','DAMSM','tripop'], default='triplet')
parser.add_argument('--feature',choices=['tensor','vector'], default='tensor')
parser.add_argument('--image_concept_file', type=str, default=None, help='Text file of image concepts in each image-caption pair')
parser.add_argument('--nfolds', type=int, default=1, help='Number of folds for cross validation')
parser.add_argument('--worker',type=int, default=0)
parser.add_argument('--only_eval',type=bool, default=False)
parser.add_argument('--alignment_scores', type=str, default=None)
parser.add_argument('--precompute_acoustic_feature', action='store_true')
parser.add_argument('--audio_model_file', type=str, default=None)
parser.add_argument('--image_model_file', type=str, default=None)
args = parser.parse_args()

if args.dataset == 'mscoco':
  if args.precompute_acoustic_feature:
    train_loader = torch.utils.data.DataLoader(
      dataloaders.ImageAudioCaptionDataset(args.data_dir,'train',
                                           max_nregions=10,
                                           image_feat_type='rcnn'),
      batch_size=args.batch_size,drop_last=True ,shuffle=True, num_workers=args.worker, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
      dataloaders.ImageAudioCaptionDataset(args.data_dir, 'val',
                                           max_nregions=10,
                                           image_feat_type='rcnn'),
      batch_size=args.batch_size, shuffle=False, num_workers=args.worker, pin_memory=True)
  else:
    audio_root_path_train = os.path.join(args.data_dir, 'train2014/wav/')
    image_root_path_train = os.path.join(args.data_dir, 'train2014/imgs/')
    segment_file_train = os.path.join(args.data_dir, 'train2014/mscoco_train_word_segments.txt')
    bbox_file_train = os.path.join(args.data_dir, 'train2014/mscoco_train_rcnn_feature.npz')
    audio_root_path_test = os.path.join(args.data_dir, 'val2014/wav/') 
    image_root_path_test = os.path.join(args.data_dir, 'val2014/imgs/')
    segment_file_test = os.path.join(args.data_dir, 'val2014/mscoco_val_word_segments.txt')
    bbox_file_test = os.path.join(args.data_dir, 'val2014/mscoco_val_rcnn_feature.npz')

    split_file = os.path.join(args.data_dir, 'val2014/mscoco_val_split.txt')
    train_loader = torch.utils.data.DataLoader(
      dataloaders.OnlineImageAudioCaptionDataset(audio_root_path_train,
                                           image_root_path_train,
                                           segment_file_train,
                                           bbox_file_train,
                                           configs={}),
      batch_size=args.batch_size, shuffle=True, num_workers=args.worker, pin_memory=True)
    
    val_loader = torch.utils.data.DataLoader(
      dataloaders.OnlineImageAudioCaptionDataset(audio_root_path_test,
                                           image_root_path_test,
                                           segment_file_test,
                                           bbox_file_test,
                                           keep_index_file=split_file,
                                           configs={}),
      batch_size=args.batch_size, shuffle=False, num_workers=args.worker, pin_memory=True)
else:
  args.data_dir = "/ws/ifp-53_2/hasegawa/lwang114/data/flickr30k/"
  train_loader = torch.utils.data.DataLoader(
    dataloaders.ImagePhoneCaptionFlickrDataset(args.data_dir,'train',
                                       max_nregions=15,
                                       image_feat_type='rcnn'),
    batch_size=args.batch_size,drop_last=True ,shuffle=True, num_workers=args.worker, pin_memory=True)
  val_loader = torch.utils.data.DataLoader(
    dataloaders.ImagePhoneCaptionFlickrDataset(args.data_dir, 'val',
                                       max_nregions=15,
                                       image_feat_type='rcnn'),
    batch_size=args.batch_size, shuffle=False, num_workers=args.worker, pin_memory=True)
  
args.exp_dir = os.path.join(args.exp_dir,args.feature,args.losstype)   

if not os.path.exists(args.exp_dir):
  os.makedirs("%s/models" % args.exp_dir)

if args.precompute_acoustic_feature:
  audio_model = models.NoOpEncoder(embedding_dim=1000)
  image_model = models.LinearTrans(input_dim=2048, embedding_dim=1000)
  attention_model = models.DotProductAttention(in_size=1000)
  if not args.only_eval:
    train_attention(audio_model, image_model, attention_model, train_loader, val_loader, args)
  else:
    evaluation_attention(audio_model, image_model, attention_model, val_loader, args)
else:
  audio_model = models.Davenet(embedding_dim=1024)
  image_model = models.LinearTrans(input_dim=2048, embedding_dim=1024)
  attention_model = models.DotProductAttention(in_size=1024)
  if not args.only_eval:
    train_attention(audio_model, image_model, attention_model, train_loader, val_loader, args)
  else: 
    loader_for_alignment = torch.utils.data.DataLoader(
      dataloaders.OnlineImageAudioCaptionDataset(audio_root_path_test,
                                                 image_root_path_test,
                                                 segment_file_test,
                                                 bbox_file_test,
                                                 keep_index_file=split_file,
                                                 configs={'return_boundary':True}),
      batch_size=args.batch_size, shuffle=False, num_workers=args.worker, pin_memory=True)

    if os.path.isdir(args.audio_model_file):
      model_dir = args.audio_model_file
      model_files = os.listdir(args.audio_model_file)
      audio_model_files = []
      image_model_files = []
      for model_file in model_files:
        if ('audio_model' in model_file) and (not 'best' in model_file):
          if int(model_file.split('.')[-2]) % 5 == 0:
            audio_model_files.append(model_file)
        elif ('image_model' in model_file) and (not 'best' in model_file):
          if int(model_file.split('.')[-2]) % 5 == 0:
            image_model_files.append(model_file)
      audio_model_files = sorted(audio_model_files, key=lambda x:int(x.split('.')[-2]))
      image_model_files = sorted(image_model_files, key=lambda x:int(x.split('.')[-2]))
      
      for audio_model_file, image_model_file in zip(audio_model_files, image_model_files):
        print(audio_model_file)
        args.audio_model_file = os.path.join(model_dir, audio_model_file)
        args.image_model_file = os.path.join(model_dir, image_model_file)
        args.alignment_scores = '{}_epoch{}'.format(args.alignment_scores, audio_model_file.split('.')[-2])
        align_attention(audio_model, image_model, loader_for_alignment, args)
    else:    
      align_attention(audio_model, image_model, loader_for_alignment, args) 
    # evaluation_attention(audio_model, image_model, attention_model, train_loader, val_loader, args)

