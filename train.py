# ==============================================================================
# Copyright (c) 2020, Yamagishi Laboratory, National Institute of Informatics
# Original: https://github.com/mkotha/WaveRNN
# Modified: Yi Zhao (zhaoyi[at]nii.ac.jp)
# All rights reserved.
# ==============================================================================
#
#
import math, pickle, os
import numpy as np
import torch
from torch.autograd import Variable
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from utils import *
import sys
import models.nocond as nc
import models.vqvae as vqvae
import models.wavernn1 as wr
import utils.env as env
import argparse
import platform
import re
import utils.logger as logger
import time
import subprocess
import pytorch_warmup as warmup
import config
import models.vqvae_f0 as vqvae_f0

parser = argparse.ArgumentParser(description='Train or run some neural net')
parser.add_argument('--generate', '-g', action='store_true')
parser.add_argument('--float', action='store_true')
parser.add_argument('--half', action='store_true')
parser.add_argument('--load', '-l')
parser.add_argument('--scratch', action='store_true')
parser.add_argument('--model', '-m')
parser.add_argument('--force', action='store_true', help='skip the version check')
parser.add_argument('--count', '-c', type=int, default=1, help='size of the test set')
parser.add_argument('--partial', action='append', default=[], help='model to partially load')
args = parser.parse_args()

if args.float and args.half:
    sys.exit('--float and --half cannot be specified together')

if args.float:
    use_half = False
elif args.half:
    use_half = True
else:
    use_half = False

model_type = args.model or 'vqvae'

model_name = f'{model_type}.43.upconv'

if model_type == 'vqvae':
    model_fn = lambda dataset: vqvae.Model(rnn_dims=config.rnn_dims, fc_dims=config.fc_dims, global_decoder_cond_dims=dataset.num_speakers(),
                  upsample_factors=config.upsample_factors, normalize_vq=True, noise_x=True, noise_y=True).cuda()
    dataset_type = 'multi'
elif model_type == 'wavernn':
    model_fn = lambda dataset: wr.Model(rnn_dims=config.rnn_dims, fc_dims=config.fc_dims, pad=2,
                  upsample_factors=config.upsample_factors, feat_dims=80).cuda()
    dataset_type = 'single'
elif model_type == 'nc':
    model_fn = lambda dataset: nc.Model(rnn_dims=896, fc_dims=896).cuda()
    dataset_type = 'single'

elif model_type == 'vcf0':
    global_decoder_cond_dims = config.dim_speaker_embedding + config.dim_gender_code + config.dim_emotion_code
    model_fn = lambda dataset: vqvae_f0.Model(rnn_dims=config.rnn_dims, fc_dims=config.fc_dims, global_decoder_cond_dims=global_decoder_cond_dims,
                  upsample_factors=config.upsample_factors, normalize_vq=True, noise_x=True, noise_y=True).cuda()
    dataset_type = 'emo'



else:
    sys.exit(f'Unknown model: {model_type}')

if dataset_type == 'multi':
    data_path = config.multi_speaker_data_path
    with open(f'{data_path}/index.pkl', 'rb') as f:
        index = pickle.load(f)
    test_index = [x[-1:] if i < 2 * args.count else [] for i, x in enumerate(index)]
    train_index = [x[:-1] if i < args.count else x for i, x in enumerate(index)]
    dataset = env.MultispeakerDataset(train_index, data_path)
elif dataset_type == 'single':
    data_path = config.single_speaker_data_path
    with open(f'{data_path}/dataset_ids.pkl', 'rb') as f:
        index = pickle.load(f)
    test_index = index[-args.count:]
    train_index = index[:-args.count]
    dataset = env.AudiobookDataset(train_index, data_path)
elif dataset_type == 'emo':
    data_path = config.emo_data_path
    # with open(f'{data_path}/dataset_ids.pkl', 'rb') as f:
    #     index = pickle.load(f)
    # test_index = index[-args.count:]
    # train_index = index[:-args.count]
    with open(f'{data_path}/index.pkl', 'rb') as f:
        index = pickle.load(f)
    test_index = index[-args.count:]
    train_index = index[:-args.count]
    gender_emo_path = config.gender_emo_path
    spk_emd_path = config.spk_emd_path
    dataset = env.EmospeakerDataset(train_index, data_path, gender_emo_path, spk_emd_path)
else:
    raise RuntimeError('bad dataset type')

print(f'dataset size: {len(dataset)}')
###############update line##################
model = model_fn(dataset)

if use_half:
    model = model.half()

for partial_path in args.partial:
    model.load_state_dict(torch.load(partial_path), strict=False)

paths = env.Paths(model_name, data_path)

if args.scratch or args.load == None and not os.path.exists(paths.model_path()):
    # Start from scratch
    step = 0
else:
    if args.load:
        prev_model_name = re.sub(r'_[0-9]+$', '', re.sub(r'\.pyt$', '', os.path.basename(args.load)))
        prev_model_basename = prev_model_name.split('_')[0]
        model_basename = model_name.split('_')[0]
        if prev_model_basename != model_basename and not args.force:
            sys.exit(f'refusing to load {args.load} because its basename ({prev_model_basename}) is not {model_basename}')
        if args.generate:
            paths = env.Paths(prev_model_name, data_path)
        prev_path = args.load
    else:
        prev_path = paths.model_path()
    step = env.restore(prev_path, model)

#model.freeze_encoder()
optimiser = optim.AdamW(model.parameters(), betas=(0.9, 0.999), weight_decay=0.01)


if args.generate:
    model.do_generate(paths, step, data_path, test_index, use_half=use_half, verbose=True)#, deterministic=True)
else:
    logger.set_logfile(paths.logfile_path())
    logger.log('------------------------------------------------------------')
    logger.log('-- New training session starts here ------------------------')
    logger.log(time.strftime('%c UTC', time.gmtime()))
    model.do_train(paths, dataset, optimiser, epochs=config.num_epochs, batch_size=config.batch_size, step=step, lr=config.lr, use_half=use_half, valid_index=test_index)
