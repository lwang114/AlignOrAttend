import argparse
import os

from models.AudioModels import *
from models.ImageModels import *
from models.SegmentModels import NoopSegmenter
from models.AlignmentLogLikelihoods import *
from steps.traintest import train, validate  
from dataloaders.image_audio_caption_dataset import ImageAudioCaptionDataset

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--exp_dir', '-e', type=str, help='Experimental directory')
parser.add_argument('--dataset', choices={'mscoco2k', 'mscoco20k', 'mscoco'})
parser.add_argument('--audio_model', choices={'lstm', 'tdnn', 'transformer'}, default='lstm')
parser.add_argument('--image_model', choices={'res34', 'vgg16'}, default='res34')
parser.add_argument('--alignment_model', choices={'mixture_aligner'}, default='mixture_aligner')
parser.add_argument('--translate_direction', choices={'sp2im', 'im2sp'}, default='sp2im', help='Translation direction (target -> source)') # TODO
parser.add_argument('--batch_size', '-b', type=int, default=16, help='Batch size')
parser.add_argument('--optim', choices={'sgd', 'adam'}, default='sgd', help='Type of optimizer')
parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
parser.add_argument('--lr-decay', type=int, default=40, help='Divide the learning rate by 10 every lr_decay epochs')
parser.add_argument('--weight-decay', type=float, default=5e-7, help='Weight decay')
parser.add_argument('--momentum', type=float, default=0.9, help='Moementum')
parser.add_argument('--n_epochs', type=int, default=20, help='Number of training epochs')
parser.add_argument('--n_print_steps', type=int, default=10, help='Print statistics every n_print_steps')
parser.add_argument('--n_phone_class', type=int, default=49, help='Number of phone unit classes')
parser.add_argument('--n_concept_class', type=int, default=80, help='Number of image concept classes')
parser.add_argument('--task', choices={'alignment', 'retrieval', 'both'}, default='retrieval', help='Type of tasks to evaluate')
parser.add_argument('--start_step', type=int, default=0, help='Starting step of the experiment')
parser.add_argument('--resume', action='store_true', help='Resume the experiment')
parser.add_argument('--device', choices={'cuda:0', 'cuda:1', 'cpu'}, default='cuda:0', help='Device to use')
args = parser.parse_args()

# Load the data paths
if not os.path.isdir('data'):
  os.mkdir('data')
if not os.path.isdir(args.exp_dir):
  os.mkdir(args.exp_dir)
  
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
    else:
      raise ValueError('Data path file not found')
    json.dump(path, f, indent=4, sort_keys=True)
else:
  with open('data/{}_path.json'.format(args.dataset), 'r') as f:
    path = json.load(f) 

# XXX if args.dataset == 'mscoco2k' or args.dataset == 'mscoco20k':
#   args.n_concept_class = args.n_phone_class = 65

configs = {'max_num_regions': 5,
           'max_num_phones': 50,
           'max_num_frames': 500}
if args.audio_model == 'transformer':
  configs['n_mfcc'] = 83
if args.translate_direction == 'sp2im':
  configs['image_first'] = True
elif args.translate_direction == 'im2sp':
  configs['image_first'] = False
  
# Set up the dataloaders
if args.audio_model == 'transformer':
  train_loader = torch.utils.data.DataLoader(
    ImageAudioCaptionDataset(path['audio_root_path_train_kaldi'], path['image_root_path_train'], path['segment_file_train'], path['bbox_file_train'], configs=configs),
    batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True
  )
  test_loader = torch.utils.data.DataLoader(
    ImageAudioCaptionDataset(path['audio_root_path_test_kaldi'], path['image_root_path_test'], path['segment_file_test'], path['bbox_file_test'], configs=configs),
    batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True
  )
else:
  train_loader = torch.utils.data.DataLoader(
    ImageAudioCaptionDataset(path['audio_root_path_train'], path['image_root_path_train'], path['segment_file_train'], path['bbox_file_train'], configs=configs),
    batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True
  )
  test_loader = torch.utils.data.DataLoader(  
    ImageAudioCaptionDataset(path['audio_root_path_test'], path['image_root_path_test'], path['segment_file_test'], path['bbox_file_test'], configs=configs),
    batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True
  )

# Initialize the image and audio encoders
if args.audio_model == 'tdnn':
  audio_model = TDNN3(n_class=args.n_phone_class)
elif args.audio_model == 'lstm':
  pretrained_model_file = '/ws/ifp-53_2/hasegawa/lwang114/summer2020/exp/blstm3_mscoco_train_sgd_lr_0.00001_mar25/audio_model.7.pth' 
  audio_model = BLSTM3(n_class=args.n_phone_class,
                       return_empty=False)
  audio_model.load_state_dict(torch.load(pretrained_model_file))
elif args.audio_model == 'transformer': # TODO Return non-empty labels only
  pretrained_model_file = '/ws/ifp-53_1/hasegawa/tools/espnet/egs/discophone/ifp_lwang114/dump/mscoco/eval/deltafalse/split1utt/data_encoder.pth'
  audio_model = Transformer(n_class=args.n_phone_class,
                            pretrained_model_file=pretrained_model_file)
  
if args.image_model == 'vgg16':
  image_model = VGG16(n_class=args.n_concept_class)
elif args.image_model == 'res34':
  image_model = Resnet34(n_class=args.n_concept_class)
  image_model.load_state_dict(torch.load('/ws/ifp-53_2/hasegawa/lwang114/fall2020/exp/res34_pretrained_model/image_model.14.pth'))

if args.alignment_model == 'mixture_aligner':
  if args.translate_direction == 'sp2im':
    alignment_model = MixtureAlignmentLogLikelihood(configs={'Ks': args.n_concept_class, 'Kt': args.n_phone_class - 1})
  elif args.translate_direction == 'im2sp':
    alignment_model = MixtureAlignmentLogLikelihood(configs={'Ks': args.n_phone_class - 1, 'Kt': args.n_concept_class})
    
# Train the model
if args.start_step <= 0:
  if args.translate_direction == 'sp2im':
    train(image_model,
          audio_model,
          NoopSegmenter({'max_nframes': configs['max_num_regions'],
                         'max_nsegments': configs['max_num_regions']}),
          NoopSegmenter({'max_nframes': configs['max_num_frames'],
                         'max_nsegments': configs['max_num_phones']}),
          alignment_model,
          train_loader,
          test_loader,
          args)
  elif args.translate_direction == 'im2sp':
    train(audio_model,
          image_model,
          NoopSegmenter({'max_nframes': configs['max_num_frames'],
                         'max_nsegments': configs['max_num_phones']}),
          NoopSegmenter({'max_nframes': configs['max_num_regions'],
                         'max_nsegments': configs['max_num_regions']}),
          alignment_model,
          train_loader,
          test_loader,
          args)

# Evaluate the model
if args.start_step <= 1:
  # TODO
  print('to-do: evaluate word discovery performance')
