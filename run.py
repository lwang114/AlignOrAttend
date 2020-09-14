import argparse
import os

from models.AudioModels import *
from models.ImageModels import *
from models.AlignmentLogLikelihoods import *
from steps.traintest import train, validate  

parser = argparse.ArgumentParser(formatter=argparse=DefaultHelpFormatter)
parser.add_argument('--exp_dir', '-e', type=str, help='Experimental directory')
parser.add_argument('--dataset', choices={'mscoco2k', 'mscoco20k', 'mscoco'})
parser.add_argument('--batch_size', '-b', type=int, default=32, help='Batch size')
parser.add_argument('--n_phone_class', type=int, default=49, help='Number of phone unit classes')
parser.add_argument('--n_concept_class', type=int, default=80, help='Number of image concept classes')
parser.add_argument('--task', choices={'alignment', 'retrieval', 'both'}, default='retrieval', help='Type of tasks to evaluate')
parser.add_argument('--start_step', type=int, default=0, help='Starting step of the experiment')
args = parser.parse_args()

# Load the data paths
if not os.path.isdir('exp'):
  os.mkdir('exp')
if not os.path.isdir('data'):
  os.mkdir('data')

if not os.path.isfile('data/{}_path.json'.format(args.dataset)):
  with open('data/{}_path.json'.format(args.dataset)) as f:
    root = '/ws/ifp-53_2/hasegawa/lwang114/data/{}'.format(args.dataset)
    if args.dataset == 'mscoco2k' or args.dataset == 'mscoco20k':
      path = {'audio_root_path_train':'{}/val2014/wav/'.format(root),
              'audio_root_path_test':'{}/val2014/wav/'.format(root),
              'segment_file_train':'{}/mscoco2k/{}_phone_info.json'.format(root, args.dataset),
              'segment_file_test':'{}/mscoco2k/{}_phone_info.json'.format(root, args.dataset),
              'image_root_path_train':'{}/val2014/imgs/val2014/'.format(root),
              'image_root_path_test':'{}/val2014/imgs/val2014/'.format(root),
              'bbox_file_train':'{}/mscoco2k/{}_bboxes.txt'.format(root, args.dataset),
              'bbox_file_train':'{}/mscoco2k/{}_bboxes.txt'.format(root, args.dataset)}
    json.dump(path, f)
else:
  with open('data/{}_path.json', 'r') as f:
    path = json.load(f) 

# Set up the dataloaders  
train_loader = torch.utils.data.DataLoader(
  dataloaders.ImageAudioCaptionDataset(path['audio_root_path_train']), path['image_root_path_train'], path['segment_file_train'], path['bbox_file_train'], 
  batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True
)

test_loader = torch.utils.data.DataLoader(  
  dataloaders.ImageAudioCaptionDataset(path['audio_root_path_test']), path['image_root_path_test'], path['segment_file_test'], path['bbox_file_test'],
  batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True
)

# Initialize the image and audio encoders
if args.audio_model == 'lstm': # TODO Load pretrained model
  audio_model = BLSTM3(n_class=args.n_phone_class)
# elif args.audio_model == 'transformer': # TODO
  
if args.image_model == 'vgg16':
  image_model = VGG16(n_class=args.n_concept_class)
elif args.image_model == 'res34':
  image_model = Resnet34(n_class=args.n_concept_class)

if args.alignment_model == 'mixture_aligner':
  alignment_model = MixtureAlignmentLogLikelihood(configs={'Ks': args.n_concept_class, 'Kt': args.n_phone_class})

# Train the model
if start_step <= 0:
  train(audio_model, image_model, alignment_model, train_loader, test_loader, args)

# Evaluate the model
if start_step <= 1:
  # TODO
  print('to-do: evaluate word discovery performance')
  return
