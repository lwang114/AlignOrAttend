import torch
import torch.nn as nn
import numpy as np
import os
import models
from dataloaders.image_audio_caption_dataset_online import OnlineImageAudioCaptionDataset
from dataloaders.image_phone_caption_dataset_segmented import ImageSegmentedPhoneCaptionDataset 
import sklearn
from sklearn.decomposition import PCA

def generate_acoustic_features(audio_model, loader, out_file,
                               image_first=False,
                               max_num_units=20,
                               l=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feats = {}
    batch_size = -1
    print('Total number of batches={}'.format(len(loader)))
    for i_b, batch in enumerate(loader):
        print('Batch {}'.format(i_b))
        if image_first:
            _, inputs, _, in_boundaries = batch
        else:
            inputs, _, in_boundaries, _ = batch
        if i_b == 0:
            batch_size = inputs.size(0)
        
        inputs = inputs.to(device)
        audio_model.to(device)
        if l < 5:
            if i_b == 0:
                print('Extracting from layer {}'.format(l))
            outputs = audio_model(inputs, l=l)
        else:
            outputs = audio_model(inputs)
            
        pooling_ratio = round(inputs.size(-1) / outputs.size(-1))
        outputs = outputs.transpose(1, 2)
        # Downsample the source segmentation by pooling ratio
        B = inputs.size(0)
        mask = np.zeros((B, max_num_units, outputs.size(1)))
        lengths = np.zeros(B, dtype=np.int64)
        for b in range(B):
            if in_boundaries.dim() == 2:
                segments = np.nonzero(in_boundaries[b].cpu().numpy())[0] // pooling_ratio
                if len(segments) <= 1:
                    print('Warning: empty caption')
                    continue
                lengths[b] = len(segments) - 1
                starts, ends = segments[:-1], segments[1:]
            else:
                starts = np.nonzero(in_boundaries[b, 0].cpu().numpy())[0] // pooling_ratio
                ends = np.nonzero(in_boundaries[b, 1].cpu().numpy())[0] // pooling_ratio
                if len(starts) == 0:
                    print('Warning: empty caption')
                    continue
                lengths[b] = len(starts)
                assert len(starts) == len(ends)

            for i_seg, (start, end) in enumerate(zip(starts, ends)):
                if i_seg >= max_num_units:
                    continue
                elif end > outputs[b].size(1):
                    print('Time stamp exceeds the maximum length: {} {}'.format(end, outputs[b].size(1)))
                    end = outputs[b].size(1)
                mask[b, i_seg, start:end] = 1. / max(end - start, 1) 

        mask = torch.FloatTensor(mask).to(device=device) 
        outputs = torch.matmul(mask, outputs)

        for j in range(B):
            arr_key = 'arr_{}'.format(i_b * batch_size + j)
            print(arr_key, outputs.size(), lengths[j])
            if lengths[j] == 0:
              feats[arr_key] = np.zeros((1, outputs[j].size(-1)))
            feats[arr_key] = outputs[j, :lengths[j]].cpu().detach().numpy()
    np.savez(out_file, **feats)

def compress_acoustic_features(feat_file,
                               compressed_dim,
                               out_file,
                               component_file=None,
                               max_nwords=20):
    feat_npz = np.load(feat_file)
    compressed_feats = {}
    keys = sorted(feat_npz, key=lambda x:int(x.split('_')[-1])) # XXX
    
    if not component_file:
        X = np.concatenate([feat_npz[k][:max_nwords] for k in keys], axis=0)
        V = PCA(n_components=compressed_dim).fit(X).components_
        print('V.shape: {}'.format(V.shape))
    compressed_feats = {k: feat_npz[k] @ V.T for k in keys} 
    np.save('{}_pca_components.npy'.format(out_file), V)
    np.savez('{}.npz'.format(out_file), **compressed_feats)
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=str, default='/ws/ifp-53_2/hasegawa/lwang114/data/mscoco/')
    parser.add_argument('--exp_dir', '-e', type=str, default='./')
    parser.add_argument('--batch_size', '-b', type=int, default=15)
    parser.add_argument('--audio_model', choices={'davenet', 'transformer'}, default='davenet')
    parser.add_argument('--model_file', type=str, default='/ws/ifp-53_1/hasegawa/tools/espnet/egs/discophone/ifp_lwang114/magnet/Image_phone_retrieval/exp/mml_rcnn_attention_10_4_2020/tensor/mml/models/best_audio_model.pth')
    parser.add_argument('--dataset', '-d', choices={'mscoco', 'speechcoco'}, default='mscoco')
    parser.add_argument('--layer_num', '-l', type=int, choices={3, 4, 5}, default=5)
    args = parser.parse_args()
    if not os.path.isdir(args.exp_dir):
        os.mkdir(args.exp_dir)
    
    if args.dataset == 'speechcoco':
        audio_root_path_train = os.path.join(args.data_dir, 'train2014/wav/')
        image_root_path_train = os.path.join(args.data_dir, 'train2014/imgs/')
        segment_file_train = os.path.join(args.data_dir, 'train2014/mscoco_train_word_segments.txt')
        bbox_file_train = os.path.join(args.data_dir, 'train2014/mscoco_train_rcnn_feature.npz')
        audio_root_path_test = os.path.join(args.data_dir, 'val2014/wav/') 
        image_root_path_test = os.path.join(args.data_dir, 'val2014/imgs/')
        segment_file_test = os.path.join(args.data_dir, 'val2014/mscoco_val_word_segments.txt')
        bbox_file_test = os.path.join(args.data_dir, 'val2014/mscoco_val_rcnn_feature.npz')
        split_file = os.path.join(args.data_dir, 'val2014/mscoco_val_split.txt')
    
        if args.audio_model == 'davenet':
            audio_model = nn.DataParallel(models.Davenet(embedding_dim=1024))
        elif args.audio_model == 'transformer':
            audio_model = nn.DataParallel(models.Transformer(embedding_dim=1024))
    
        train_set = OnlineImageAudioCaptionDataset(audio_root_path_train,
                                         image_root_path_train,
                                         segment_file_train,
                                         bbox_file_train,
                                         configs={'return_boundary': True})
        test_set = OnlineImageAudioCaptionDataset(audio_root_path_test,
                                        image_root_path_test,
                                        segment_file_test,
                                        bbox_file_test,
                                        keep_index_file=split_file, 
                                        configs={'return_boundary': True})
    elif args.dataset == 'mscoco':
        segment_file_train = os.path.join(args.data_dir, 'train2014/mscoco_train_word_phone_segments.txt')
        segment_file_test = os.path.join(args.data_dir, 'val2014/mscoco_val_word_phone_segments.txt') # TODO
        bbox_file_train = os.path.join(args.data_dir, 'train2014/mscoco_train_rcnn_feature.npz')
        bbox_file_test = os.path.join(args.data_dir, 'val2014/mscoco_val_rcnn_feature.npz')
        split_file = os.path.join(args.data_dir, 'val2014/mscoco_val_split.txt')
    
        if args.audio_model == 'davenet':
            audio_model = nn.DataParallel(models.DavenetSmall(input_dim=49, embedding_dim=512))
        elif args.audio_model == 'transformer':
            raise NotImplementedError
    
        test_set = ImageSegmentedPhoneCaptionDataset(args.data_dir,
                                        segment_file_test,
                                        split='val')
        train_set = ImageSegmentedPhoneCaptionDataset(args.data_dir,
                                         segment_file_train)
        

    audio_model.load_state_dict(torch.load(args.model_file))
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=args.batch_size, 
                                               shuffle=False, 
                                               num_workers=1, 
                                               pin_memory=True)
    
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=args.batch_size, 
                                              shuffle=False, 
                                              num_workers=1, 
                                              pin_memory=True)
    ''' 
    generate_acoustic_features(audio_model,
                               train_loader,
                               out_file='{}/train_features.npz'.format(args.exp_dir))
    '''
    generate_acoustic_features(audio_model,
                               test_loader,
                               out_file='{}/test_features.npz'.format(args.exp_dir))
    '''
    compress_acoustic_features('{}/train_features.npz'.format(args.exp_dir),
                               compressed_dim=300, # XXX
                               out_file='{}/train_features_300dim'.format(args.exp_dir))
    '''
    compress_acoustic_features('{}/test_features.npz'.format(args.exp_dir),
                               compressed_dim=300, # XXX
                               out_file='{}/test_features_300dim'.format(args.exp_dir))
                               # component_file='{}/train_features_300dim_pca_components.npy'.format(args.exp_dir))
    '''
    compress_acoustic_features('{}/train2014/mscoco_train_ctc_embed1000dim_word.npz'.format(args.data_dir),
                               compressed_dim=300, # XXX
                               out_file='{}/train_features_300dim'.format(args.exp_dir))
    compress_acoustic_features('{}/val2014/mscoco_val_ctc_embed1000dim_word.npz'.format(args.data_dir),
                               compressed_dim=300, # XXX
                               out_file='{}/test_features_300dim'.format(args.exp_dir),
                               component_file='{}/train_features_300dim_pca_components.npy'.format(args.exp_dir))
    
    compress_acoustic_features('{}/train2014/mscoco_train_rcnn_feature.npz'.format(args.data_dir),
                               compressed_dim=512, # XXX
                               out_file='{}/train_features_512dim'.format(args.exp_dir),
                               max_nwords=10)
    compress_acoustic_features('{}/val2014/mscoco_val_rcnn_feature.npz'.format(args.data_dir),
                               compressed_dim=512, # XXX
                               out_file='{}/test_features_512dim'.format(args.exp_dir),
                               component_file='{}/train_features_512dim_pca_components.npy'.format(args.exp_dir),
                               max_nwords=10)
    '''
