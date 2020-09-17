import argparse
import os

from models.AudioModels import *
from models.ImageModels import *
from models.AlignmentLogLikelihoods import *
from steps.traintest import train, validate  
from dataloaders.image_audio_caption_dataset import ImageAudioCaptionDataset

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--exp_dir', '-e', type=str, help='Experimental directory')
parser.add_argument('--dataset', choices={'mscoco2k', 'mscoco20k', 'mscoco'})
parser.add_argument('--audio_model', choices={'lstm', 'tdnn', 'transformer'}, default='lstm')
parser.add_argument('--image_model', choices={'res34', 'vgg16'}, default='res34')
parser.add_argument('--alignment_model', choices={'mixture_aligner'}, default='mixture_aligner')
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
parser.add_argument('--device', choices={'gpu', 'cpu'}, default='gpu', help='Device to use')
args = parser.parse_args()

# Load the data paths
if not os.path.isdir('exp'):
  os.mkdir('exp')
if not os.path.isdir('data'):
  os.mkdir('data')

if not os.path.isfile('data/{}_path.json'.format(args.dataset)):
  with open('data/{}_path.json'.format(args.dataset), 'w') as f:
    root = '/ws/ifp-53_2/hasegawa/lwang114/data/mscoco/'
    if args.dataset == 'mscoco2k':
      path = {'root': root,
              'audio_root_path_train':'{}/mscoco2k/wav/'.format(root),
              'audio_root_path_test':'{}/mscoco2k/wav/'.format(root),
              'segment_file_train':'{}/mscoco2k/{}_phone_info.json'.format(root, args.dataset),
              'segment_file_test':'{}/mscoco2k/{}_phone_info.json'.format(root, args.dataset),
              'image_root_path_train':'{}/val2014/imgs/val2014/'.format(root),
              'image_root_path_test':'{}/val2014/imgs/val2014/'.format(root),
              'bbox_file_train':'{}/mscoco2k/{}_bboxes.txt'.format(root, args.dataset),
              'bbox_file_test':'{}/mscoco2k/{}_bboxes.txt'.format(root, args.dataset)}
    else:
      raise NotImplementedError
    json.dump(path, f)
else:
  with open('data/{}_path.json'.format(args.dataset), 'r') as f:
    path = json.load(f) 

configs = {}
if args.audio_model == 'transformer':
  configs = {'n_mfcc': 83}
  
# Set up the dataloaders  
train_loader = torch.utils.data.DataLoader(
  ImageAudioCaptionDataset(path['audio_root_path_train'], path['image_root_path_train'], path['segment_file_train'], path['bbox_file_train'], configs=configs),
  batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True
)

test_loader = torch.utils.data.DataLoader(  
  ImageAudioCaptionDataset(path['audio_root_path_test'], path['image_root_path_test'], path['segment_file_test'], path['bbox_file_test'], configs=configs),
  batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True
)

# Initialize the image and audio encoders
if args.audio_model == 'tdnn':
  audio_model = TDNN3(n_class=args.n_phone_class)
elif args.audio_model == 'lstm':
  audio_model = BLSTM3(n_class=args.n_phone_class)
  audio_model.load_state_dict(torch.load('/ws/ifp-53_2/hasegawa/lwang114/summer2020/exp/blstm3_mscoco_train_sgd_lr_0.00001_mar25/audio_model.7.pth'))
elif args.audio_model == 'transformer': # TODO Add option to specify the pretrained model file
  audio_model = Transformer(n_class=args.n_phone_class)
  #                          pretrained_model_file='/ws/ifp-53_1/hasegawa/tools/espnet/egs/discophone/ifp_lwang114/dump/mscoco/eval/deltafalse/split1utt/data_encoder.pth')
  
if args.image_model == 'vgg16':
  image_model = VGG16(n_class=args.n_concept_class)
elif args.image_model == 'res34':
  image_model = Resnet34(n_class=args.n_concept_class)

if args.alignment_model == 'mixture_aligner':
  alignment_model = MixtureAlignmentLogLikelihood(configs={'Ks': args.n_concept_class, 'Kt': args.n_phone_class})

# Train the model
if args.start_step <= 0:
  train(audio_model, image_model, alignment_model, train_loader, test_loader, args)

# Evaluate the model
if args.start_step <= 1:
  # TODO
  print('to-do: evaluate word discovery performance')
