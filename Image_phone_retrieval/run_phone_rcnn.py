# Modified from https://github.com/dharwath/DAVEnet-pytorch.git
import argparse
import os
import pickle
import sys
import time
import torch
import dataloaders
import models
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
parser.add_argument("--simtype", type=str, default="MISA",
        help="matchmap similarity function", choices=["SISA", "MISA", "SIMA"])
parser.add_argument('--losstype', choices=['triplet', 'mml','DAMSM','tripop'], default='triplet')
parser.add_argument('--feature',choices=['tensor','vector'], default='tensor')
parser.add_argument('--image_concept_file', type=str, default=None, help='Text file of image concepts in each image-caption pair')
parser.add_argument('--nfolds', type=int, default=1, help='Number of folds for cross validation')
parser.add_argument('--worker',type=int, default=8)
parser.add_argument('--only_eval',type=bool, default=False)
parser.add_argument('--alignment_scores', type=str, default=None)
args = parser.parse_args()

if args.dataset == 'mscoco':
  train_loader = torch.utils.data.DataLoader(
    dataloaders.ImagePhoneCaptionDataset(args.data_dir,'train',
                                       max_nregions=10,
                                       image_feat_type='rcnn'),
    batch_size=args.batch_size,drop_last=True ,shuffle=True, num_workers=args.worker, pin_memory=True)
  val_loader = torch.utils.data.DataLoader(
    dataloaders.ImagePhoneCaptionDataset(args.data_dir, 'val',
                                       max_nregions=10,
                                       image_feat_type='rcnn'),
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

if args.feature == 'tensor':
  if args.dataset == 'mscoco':
    input_dim = 49
  else:
    input_dim = 81
    
  audio_model = models.DavenetSmall(input_dim=input_dim, embedding_dim=512)
  image_model = models.LinearTrans(input_dim=2048, embedding_dim=512)
  if not args.only_eval:
    train(audio_model, image_model, train_loader, val_loader, args)
  else:
    evaluation(audio_model,image_model,val_loader,args)
elif args.feature == 'vector':
  audio_model = models.CNN_RNN_ENCODER(embedding_dim=512,n_layer=3)
  image_model = models.LinearTrans(embedding_dim=512)
  if not args.only_eval:
    train(audio_model, image_model, train_loader, val_loader, args)
  else:
    evaluation_vector(audio_model,image_model,val_loader,args)
  



