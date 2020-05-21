# ==============================================================================
# Copyright (c) 2020, Yamagishi Laboratory, National Institute of Informatics
# Author: Yi Zhao (zhaoyi@nii.ac.jp)
# All rights reserved.
# ==============================================================================
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
import models.f0encode as f0encode
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

global_decoder_cond_dims = config.dim_speaker_embedding + config.dim_gender_code + config.dim_emotion_code
model_fn = vqvae_f0.Model(rnn_dims=config.rnn_dims, fc_dims=config.fc_dims, global_decoder_cond_dims=global_decoder_cond_dims,
              upsample_factors=config.upsample_factors, normalize_vq=True, noise_x=True, noise_y=True).cuda()


model_f0 = f0encode.Model(upsample_factors=config.upsample_factors, normalize_vq=True).cuda()


step=0
test_files_txt = config.test_files_txt

output_dir = config.test_output_dir

with open(test_files_txt) as f:
    lines = f.readlines()



lines = [x.strip() for x in lines]
total_num_files = len(lines)
test_index = []
target_index = []
target_emotion = []
emo_dict = {'neu': [0,0,0,1], 'ang':[0,0,1,0], 'joy':[0,1,0,0], 'sad':[1,0,0,0] }

for line in lines:
    test_index.append(line.split(" ")[0])
    target_index.append(line.split(" ")[1])
    target_emotion.append(emo_dict[line.split(" ")[2]])


pretrained_dict = model_fn.state_dict()
model_dict = model_f0.state_dict()

# 1. filter out unnecessary keys
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# 2. overwrite entries in the existing state dict
model_dict.update(pretrained_dict)
# 3. load the new state dict
model_f0.load_state_dict(pretrained_dict)

dataset = env.EmospeakerDataset(test_index, config.emo_data_path, config.gender_emo_path, config.spk_emd_path)
loader = DataLoader(dataset, shuffle=False)
data = [x for x in loader]

f0s = [torch.FloatTensor(f0[0].float()) for x, f0, global_cond in data]
# print(f0s)
maxlen_f0 = max([len(x) for x in f0s])
aligned_f0 = [torch.cat([torch.FloatTensor(x), torch.zeros(maxlen_f0-len(x))]) for x in f0s]
f0s = torch.stack(aligned_f0 , dim=0).cuda()
out = model_f0.forward_generate(f0s)
for i, x in enumerate(f0s):
