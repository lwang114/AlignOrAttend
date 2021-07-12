#!/bin/bash

stage=1
stop_stage=100

if [ $stage -le 0 ] || [ $stop_stage -ge 0 ]; then
  # Done manually: download the rcnn features from 
  # https://storage.googleapis.com/up-down-attention/trainval_36.zip 
  # and put it under the data root specified in the config file 

  # Prepare data
  python utils/read_tsv.py config/speechcoco.json
fi

if [ $stage -le 1 ] || [ $stop_stage -ge 1 ]; then
  # Train Davenet and extract its embeddings
  cd davenet
  python run_audio_rcnn_attention.py --config ../config/davenet_speechcoco.json  
  python gen_feats.py --config ../config/davenet_speechcoco.json
  cd ../
fi

if [ $stage -le 2 ] || [ $stop_stage -ge 2 ]; then
  # Train DNN HMM DNN
  cd dnn_hmm_dnn
  python fully_continuous_mixture_aligner.py ../config/speechcoco.json
  cd ../
fi

if [ $stage -le 3 ] || [ $stop_stage -ge 3 ]; then
  # Evaluation
  python utils/evaluate_tde2.py config/speechcoco.json --segment --task 0
  python utils/evaluate_tde2.py config/speechcoco.json --segment --task 1
  python utils/evaluate_tde2.py config/speechcoco.json --segment --task 2
fi

