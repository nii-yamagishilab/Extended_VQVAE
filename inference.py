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

import models.wavernn1 as wr
import config
import librosa


if __name__ == "__main__":
    test_files_txt = config.test_files_txt

    output_dir = config.test_output_dir
    os.makedirs(output_dir, exist_ok=True)

    model_path = config.test_model_path
    sample_rate = config.sample_rate
    batch_size = config.batch_size


    model = wr.Model(rnn_dims=config.rnn_dims, fc_dims=config.fc_dims, pad=2,
                  upsample_factors=config.upsample_factors, feat_dims=80).cuda()
    model.load_state_dict(torch.load(model_path))

    with open(test_files_txt) as f:
        test_files = f.readlines()
    # print(test_files)
    test_files = [x.strip() for x in test_files]
    total_num_files = len(test_files)
    print('number of files: %d' %(total_num_files))
    for i in range(int(total_num_files/batch_size) + 1):
        if (i+1) * batch_size <= total_num_files:
            files = test_files[(i*batch_size) : ((i+1) * batch_size)]
        else:
            files = test_files[(i*batch_size) : total_num_files]
        if len(files) == 0:
            print('no more files')
            exit ()
        test_mels = [np.load(x) for x in files]
        maxlen = max([x.shape[1] for x in test_mels])
        len_test_mels = [x.shape[1] for x in test_mels]
        aligned = [torch.cat([torch.FloatTensor(x), torch.zeros(80, maxlen-x.shape[1]+1)], dim=1) for x in test_mels]
        #print(torch.stack(aligned).size())
        out = model.forward_generate((torch.stack(aligned)).cuda(), deterministic=True)
        for j in range(len(files)):
            audio = out[j][:((len_test_mels[j]-3)*config.hop_length)].cpu().numpy()
            #print(audio.shape)
            basename = os.path.basename(files[j]).split('.')[0]
            librosa.output.write_wav(os.path.join(output_dir, basename+'.wav'), audio, sr = sample_rate)
