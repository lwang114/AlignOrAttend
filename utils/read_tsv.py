#!/usr/bin/env python
# Utility to read .tsv file from https://github.com/peteanderson80/bottom-up-attention.git

import base64
import numpy as np
import csv
import sys
import zlib
import time
import mmap
import json
import os
import argparse

csv.field_size_limit(sys.maxsize)
   
FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']

if __name__ == '__main__':
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('CONFIG', type=str)
  args = parser.parse_args() 

  config = json.load(open(args.config)) 
  root = config.root
  infile = '{}/trainval_36/trainval_resnet101_faster_rcnn_genome_36.tsv'.format(root)
  for x in ['train', 'val']:
    feat_root = '{}/{}2014/rcnn_feats/'.format(root, x)
    if not os.path.isdir(feat_root):
        os.mkdir(feat_root)
    bbox_file = '{}/{}2014/mscoco_{}_bboxes.txt'.format(root, x, x)
    bbox_outfile = '{}/{}2014/mscoco_{}_bboxes_rcnn.json'.format(root, x, x)
    
    # TODO Read a list of image ids
    img_ids = [] 
    img_id = ''
    with open(bbox_file, 'r') as f:
      for line in f:
        cur_img_id = line.split()[0]
        if cur_img_id != img_id:
          img_ids.append(cur_img_id)
          img_id = cur_img_id
    
    # Verify we can read a tsv
    in_data = {}
    with open(infile, "r+b") as tsv_in_file,\
         open(bbox_outfile, 'w') as json_out_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
        i_item = 0
        for item in reader:
            for ex, img_id in enumerate(img_ids):
                if int(img_id.split('_')[-1]) == int(item['image_id']):
                    break

            arr_key = '{}_{}'.format(img_id, ex)
            print('Instance {} with image id key {}'.format(i_item, arr_key))
            # if i_item > 30:
            #     break
            i_item += 1
            item['image_id'] = int(item['image_id'])
            item['image_h'] = int(item['image_h'])
            item['image_w'] = int(item['image_w'])   
            item['num_boxes'] = int(item['num_boxes'])
            for field in ['boxes', 'features']:
                item[field] = np.frombuffer(base64.decodestring(item[field]), 
                      dtype=np.float32).reshape((item['num_boxes'],-1))

            np.save('{}/{}.npy'.format(feat_root, arr_key), item['features'])
            in_data[arr_key] = item['boxes'].tolist()
            # break
        json.dump(in_data, json_out_file, indent=4, sort_keys=True)
    # print in_data
