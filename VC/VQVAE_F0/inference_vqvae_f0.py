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
import soundfile as sf
import models.wavernn1 as wr
import config
import librosa
import models.vqvae_f0 as vqvae_f0
#paths, step, data_path, test_index, deterministic=False, use_half=False, verbose=False
def do_generate(model, gen_path, step, data_path, test_index, target_index, target_emotion, deterministic=False, use_half=False, verbose=False):
    k = step//1000
    print(test_index)
    dataset = env.EmospeakerDataset(test_index, data_path, config.gender_emo_path, config.spk_emd_path)
    loader = DataLoader(dataset, shuffle=False)
    data = [x for x in loader]
    n_points = len(data)
    gt = [(x[0].float() + 0.5) / (2**15 - 0.5) for x, f0, global_cond in data]
    extended = [np.concatenate([np.zeros(model.pad_left_encoder(), dtype=np.float32), x, np.zeros(model.pad_right(), dtype=np.float32)]) for x in gt]
    maxlen = max([len(x) for x in extended])
    aligned = [torch.cat([torch.FloatTensor(x), torch.zeros(maxlen-len(x))]) for x in extended]

    f0s = [torch.FloatTensor(f0[0].float()) for x, f0, global_cond in data]
    maxlen_f0 = max([len(x) for x in f0s])
    aligned_f0 = [torch.cat([torch.FloatTensor(x), torch.zeros(maxlen_f0-len(x))]) for x in f0s]
    global_conds = [torch.FloatTensor(global_cond[0].float()) for x, f0, global_cond in data]


    target_dataset = env.EmospeakerDataset(target_index, data_path, config.gender_emo_path, config.spk_emd_path)
    target_loader = DataLoader(target_dataset,   shuffle=False)
    target_data = [x for x in target_loader]
    target_global_conds = [global_cond[0].float() for x, f0, global_cond in target_data]
    #global_cond = gd*5 + emod*10 + spkd-> 10:50
    # print(len(target_global_conds), len(target_global_conds[0]))
    for i in range(len(target_emotion)):
        target_global_conds[i][10:50]= torch.FloatTensor(target_emotion[i]*10)

    # print(target_global_conds, target_emotion)
    #target_global_conds[:, 10:50] = torch.from_numpy(target_emotion)

    # target_global_conds = torch.FloatTensor(target_global_conds).cuda()

    os.makedirs(gen_path, exist_ok=True)
    #global_decoder_cond, samples, f0,
    #conds = torch.stack(global_conds + target_global_conds, dim=0).cuda()
    conds = torch.stack(target_global_conds, dim=0).cuda()
    audios = torch.stack(aligned , dim=0).cuda()
    f0s = torch.stack(aligned_f0 , dim=0).cuda()

    #audios = torch.stack(aligned + aligned, dim=0).cuda()

    #f0s = torch.stack(aligned_f0 + aligned_f0, dim=0).cuda()

    out = model.forward_generate(conds,audios , f0s ,verbose=verbose, use_half=use_half)
    #out = model.forward_generate(global_conds.cuda(), aligned.cuda(), f0s.cuda(), verbose=verbose, use_half=use_half)

    logger.log(f'out: {out.size()}')
    for i, x in enumerate(gt) :
        #librosa.output.write_wav( test_index[i] + '_source.wav', x.cpu().numpy(), sr=sample_rate)
        #audio = out[i][:len(x)].cpu().numpy()
        #librosa.output.write_wav(f'{gen_path}/{k}k_steps_{i}_resynthesis.wav', audio, sr=sample_rate)
        #audio_tr = out[n_points+i][:len(x)].cpu().numpy()
        #librosa.output.write_wav(f'{gen_path}/{k}k_steps_{i}_transferred.wav', audio_tr, sr=sample_rate)
        audio = out[i][:len(x)].cpu().numpy()
        sf.write(f'{gen_path}/{test_index[i]}&{target_index[i]}.wav', audio, sample_rate, subtype='PCM_16')


if __name__ == "__main__":
    test_files_txt = config.test_files_txt

    output_dir = config.test_output_dir
    os.makedirs(output_dir, exist_ok=True)

    model_path = config.test_model_path
    sample_rate = config.sample_rate
    batch_size = config.batch_size

    gender_emo_path = config.gender_emo_path
    spk_emd_path = config.spk_emd_path

    global_decoder_cond_dims = config.dim_speaker_embedding + config.dim_gender_code + config.dim_emotion_code
    model = vqvae_f0.Model(rnn_dims=config.rnn_dims, fc_dims=config.fc_dims, global_decoder_cond_dims=global_decoder_cond_dims,
                  upsample_factors=config.upsample_factors, normalize_vq=True, noise_x=True, noise_y=True).cuda()
    model.load_state_dict(torch.load(model_path))

    #gen_path, step, data_path, test_index, target_global_conds
    #step = int(model_path.split('/')[-1].split('_')[-1].split('.')[0])
    if 'vcf0' == model_path.split('/')[-1].split('_')[-1].split('.')[0]:
        step = 0
    else:
        step = int(model_path.split('/')[-1].split('_')[-1].split('.')[0])

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

    i = 0
    j = 40

    while i + j <= len(test_index):
        print(i,i+j)
        do_generate(model,output_dir, step ,config.emo_data_path, test_index[i:i+j], target_index[i:i+j], target_emotion[i:i+j])
        i += j
    if (len(test_index) - i) > 0:
        print(i)
        do_generate(model,output_dir, step ,config.emo_data_path, test_index[i:], target_index[i:], target_emotion[i:])
