import torch
from Image_phone_retrieval.steps.util import *
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import numpy as np

def plot_attention(loader, audio_model, image_model, n_examples=10, out_dir='./'): # TODO
  """ Generate the attention matrix learned by the word discovery model """
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  indices = np.arange(len(loader))
  example_ids = np.random.permutation(indices)[:n_examples]
  B = -1
  for i_b, batch in enumerate(loader):
    if i_b == 0:
      B = image_input.size(0)

    image_input, audio_input, nregions, nphones = batch
    D = image_input.size(-1)
    
    audio_output = audio_model(audio_input)
    image_output = image_model(image_input)
    
    # Compute the dot-product attention weights
    for b in range(B):
      example_id = i_b * B + b 
      if example_id in indices:
        print('Example {}'.format(example_id))
        matchmap = torch.mm(image_output, audio_output).t()
        attention = (matchmap / np.sqrt(D)).softmax(-1).cpu().numpy()  

        # Plot the attention weights
        plt.figure(size=(30, 40))
        fig, ax = plt.subplots()
        plt.imshow(attention, cmap=plt.Greys, vmin=0, vmax=1)
        # Display word-level boundary labels
        plt.savefig('{}/attention_{}.png'.format(out_dir, i_example))

def plot_boxes(loader, box_file): # TODO
  """ Generate the boxes discovered by the RCNN """
  pass

def plot_clusters(embedding_vec_file, word_segment_file, n_clusters=10): # XXX
  """ Visualize the clusters discovered by the system using t-SNE """
  pass

if __name__ == '__main__':
  import models
  from dataloader.image_audio_caption_dataset import ImageAudioCaptionDataset
  import argparse
  import os

  parser = argparse.ArgumentParser(formatter=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--exp_dir', '-e', type=str, default='./')
  args = parser.parse_args()
  
  out_dir = '{}/outputs/'.format(args.exp_dir)
  if not os.path.isfile(out_dir):
    os.mkdir(out_dir)
  data_dir = '/ws/ifp-53_2/hasegawa/lwang114/data/mscoco/' 
  loader = ImageAudioCaptionDataset(data_dir, 'val')
  audio_model = models.Davenet(embedding_dim=1024)
  image_model = models.LinearTrans(input_dim=2048, embedding_dim=1024)
  plot_attention(loader, audio_model, image_model, out_dir)
