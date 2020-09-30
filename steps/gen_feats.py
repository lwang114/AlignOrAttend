import numpy as np

def generate_acoustic_features(audio_model, segmenter, loader, configs, args, out_file):
    feats = {}
    B = args.batch_size
    print('Batch size={}, total number of batches={}'.format(B, len(loader)))
    for i_b, batch in enumerate(loader):
        if configs.get('image_first', True):
            _, inputs, _, in_boundaries = batch
        else:
            inputs, _, in_boundaries, _ = batch

        embeds, outputs = audio_model(inputs, save_features=True)
        embeds, masks, _ = segmenter(embeds, in_boundaries, is_embed=True)

        lengths = masks.sum(-1).cpu().detach().numpy().astype(int)
        for j in range(embeds.size(0)):
            arr_key = 'arr_{}'.format(i_b * B + j)
            print(arr_key)
            feats[arr_key] = embeds[j, :lengths[j]].cpu().detach().numpy()
    np.savez('{}/{}'.format(args.exp_dir, out_file), **feats)
