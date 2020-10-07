import torch
from Image_phone_retrieval.steps.util import *
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import numpy as np

def plot_attention(loader, audio_model, image_model, n_examples=10, out_dir='./'): # TODO
  """ Generate the attention matrix learned by the word discovery model """
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  indices = np.arange(1000)
  example_ids = np.random.permutation(indices)[:n_examples]
  B = -1
  for i_b, batch in enumerate(loader):
    audio_input, image_input, nphones, nregions = batch
    D = image_input.size(-1)
    if i_b == 0:
      B = image_input.size(0)
    
    # Compute the dot-product attention weights
    for b in range(B):
      example_id = i_b * B + b 
      if example_id in example_ids:
        print('Example {}'.format(example_id))
        audio_output = audio_model(audio_input)
        image_output = image_model(image_input)
        matchmap = torch.mm(image_output[b], audio_output[b]).t()
        attention = (matchmap).softmax(-1).cpu().detach().numpy()  
        
        # Plot the attention weights
        fig, ax = plt.subplots(figsize=(30, 40))
        ax.invert_yaxis()

        plt.pcolor(attention, cmap=plt.cm.Greys, vmin=attention.min(), vmax=attention.max())
        cbar = plt.colorbar()
        for tick in cbar.ax.get_yticklabels():
          tick.set_fontsize(30)

        # TODO Display region labels
        # TODO Display word-level boundary labels
        plt.savefig('{}/attention_{}.png'.format(out_dir, example_id))

def plot_boxes(loader, box_file): # TODO
  """ Generate the boxes discovered by the RCNN """
  pass

def plot_clusters(embedding_vec_file, word_segment_file, n_clusters=10): # XXX
  """ Visualize the clusters discovered by the system using t-SNE """
  pass

if __name__ == '__main__':
  import Image_phone_retrieval.models as models
  from Image_phone_retrieval.dataloaders.image_audio_caption_dataset_online import OnlineImageAudioCaptionDataset
  import argparse
  import os

  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--exp_dir', '-e', type=str, default='/ws/ifp-53_1/hasegawa/tools/espnet/egs/discophone/ifp_lwang114/magnet/Image_phone_retrieval/exp/biattentive_mml_rcnn_10_4_2020/tensor/mml')
  parser.add_argument('--data_dir', '-d', type=str, default='/ws/ifp-53_2/hasegawa/lwang114/data/mscoco/')
  args = parser.parse_args()
  
  out_dir = '{}/outputs/'.format(args.exp_dir)
  if not os.path.isdir(out_dir):
    os.mkdir(out_dir)
  audio_model_file = '{}/models/best_audio_model.pth'.format(args.exp_dir)
  image_model_file = '{}/models/best_image_model.pth'.format(args.exp_dir)

  audio_root_path_test = os.path.join(args.data_dir, 'val2014/wav/') 
  image_root_path_test = os.path.join(args.data_dir, 'val2014/imgs/')
  segment_file_test = os.path.join(args.data_dir, 'val2014/mscoco_val_word_segments.txt')
  bbox_file_test = os.path.join(args.data_dir, 'val2014/mscoco_val_rcnn_feature.npz')
  split_file = os.path.join(args.data_dir, 'val2014/mscoco_val_split.txt')
  
  loader = torch.utils.data.DataLoader(
    OnlineImageAudioCaptionDataset(audio_root_path_test,
                                          image_root_path_test,
                                          segment_file_test,
                                          bbox_file_test,
                                          keep_index_file=split_file,
                                          configs={}),
    batch_size=15, shuffle=False, num_workers=0, pin_memory=True)
  audio_model = models.Davenet(embedding_dim=1024)
  image_model = models.LinearTrans(input_dim=2048, embedding_dim=1024)

  plot_attention(loader, audio_model, image_model, out_dir=out_dir)
