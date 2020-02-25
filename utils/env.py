from torch.utils.data import Dataset
import torch
import os
import numpy as np
from utils.dsp import *
import re
import config
import pandas as pd

bits = 16
seq_len = config.hop_length * 5

class Paths:
    def __init__(self, name, data_dir, checkpoint_dir=config.checkpoint_dir, output_dir=config.output_dir):
        self.name = name
        self.data_dir = data_dir
        self.checkpoint_dir = checkpoint_dir
        self.output_dir = output_dir

    def model_path(self):
        return f'{self.checkpoint_dir}/{self.name}.pyt'

    def model_hist_path(self, step):
        return f'{self.checkpoint_dir}/{self.name}_{step}.pyt'

    def step_path(self):
        return f'{self.checkpoint_dir}/{self.name}_step.npy'

    def gen_path(self):
        return f'{self.output_dir}/{self.name}/'

    def logfile_path(self):
        return f'log/{self.name}'

def default_paths(name, data_dir):
    return Paths(name, data_dir, checkpoint_dir=config.checkpoint_dir, output_dir=config.output_dir)

class AudiobookDataset(Dataset):
    def __init__(self, ids, path):
        self.path = path
        self.metadata = ids

    def __getitem__(self, index):
        file = self.metadata[index]
        m = np.load(f'{self.path}/mel/{file}.npy')
        x = np.load(f'{self.path}/quant/{file}.npy')
        return m, x

    def __len__(self):
        return len(self.metadata)

class MultispeakerDataset(Dataset):
    def __init__(self, index, path):
        self.path = path
        self.index = index
        self.all_files = [(i, name) for (i, speaker) in enumerate(index) for name in speaker]

    def __getitem__(self, index):
        speaker_id, name = self.all_files[index]
        speaker_onehot = (np.arange(len(self.index)) == speaker_id).astype(np.long)
        audio = np.load(f'{self.path}/{speaker_id}/{name}.npy')
        return speaker_onehot, audio

    def __len__(self):
        return len(self.all_files)

    def num_speakers(self):
        return len(self.index)

class EmospeakerDataset(Dataset):
    def __init__(self, index, path, gender_emo_path, spk_emd_path):
        self.all_utts = index #train.scp include utt name of training files
        self.path = path
        self.df_gender_emo = pd.read_csv(gender_emo_path, sep="\t", dtype=object, header=0)
        self.df_spk_emd = pd.read_csv(spk_emd_path, sep="\t", dtype=object, header=0)
        self.emo_dict = {'neu': [0,0,0,1], 'ang':[0,0,1,0], 'joy':[0,1,0,0], 'sad':[1,0,0,0] }
        self.gender_dict = {'F':[1,0], 'M':[0,1]}
        self.emd_names = ['emd' + str(i) for i in range(config.dim_speaker_embedding )]


    def __getitem__(self, index):
        utt = self.all_utts[index]
        audio = np.load(f'{self.path}/quant/{utt}.npy')
        f0 = np.load(f'{self.path}/f0/{utt}.npy')
        gd = self.gender_dict[self.df_gender_emo[self.df_gender_emo['utt'] == utt]['gender'].values[-1]]
        emod = self.emo_dict[self.df_gender_emo[self.df_gender_emo['utt'] == utt]['emotion'].values[-1]]
        spk = utt.split('_')[1]
        spkd = self.df_spk_emd[self.df_spk_emd['speaker'] == spk][self.emd_names].astype(float).values.tolist()[-1]
        global_cond = gd*5 + emod*10 + spkd

        return  audio, f0, np.array(global_cond)

    def __len__(self):
        return len(self.all_utts)

def collate_multispeaker_samples(left_pad, window, right_pad, batch):
    samples = [x[1] for x in batch]
    speakers_onehot = torch.FloatTensor([x[0] for x in batch])
    max_offsets = [x.shape[-1] - window for x in samples]
    offsets = [np.random.randint(0, offset) for offset in max_offsets]

    wave16 = [np.concatenate([np.zeros(left_pad, dtype=np.int16), x, np.zeros(right_pad, dtype=np.int16)])[offsets[i]:offsets[i] + left_pad + window + right_pad] for i, x in enumerate(samples)]
    return torch.FloatTensor(speakers_onehot), torch.LongTensor(np.stack(wave16).astype(np.int64))

def collate_samples(left_pad, window, right_pad, batch):
    #print(f'collate: window={window}')
    samples = [x[1] for x in batch]
    max_offsets = [x.shape[-1] - window for x in samples]
    offsets = [np.random.randint(0, offset) for offset in max_offsets]

    wave16 = [np.concatenate([np.zeros(left_pad, dtype=np.int16), x, np.zeros(right_pad, dtype=np.int16)])[offsets[i]:offsets[i] + left_pad + window + right_pad] for i, x in enumerate(samples)]
    return torch.LongTensor(np.stack(wave16).astype(np.int64))

def collate(left_pad, mel_win, right_pad, batch) :
    max_offsets = [x[0].shape[-1] - mel_win for x in batch]
    mel_offsets = [np.random.randint(0, offset) for offset in max_offsets]
    sig_offsets = [offset * hop_length for offset in mel_offsets]

    mels = [x[0][:, mel_offsets[i]:mel_offsets[i] + mel_win] for i, x in enumerate(batch)]

    wave16 = [np.concatenate([np.zeros(left_pad, dtype=np.int16), x[1], np.zeros(right_pad, dtype=np.int16)])[sig_offsets[i]:sig_offsets[i] + left_pad + hop_length * mel_win + right_pad] for i, x in enumerate(batch)]

    mels = np.stack(mels).astype(np.float32)
    wave16 = np.stack(wave16).astype(np.int64) + 2**15
    coarse = wave16 // 256
    fine = wave16 % 256

    mels = torch.FloatTensor(mels)
    coarse = torch.LongTensor(coarse)
    fine = torch.LongTensor(fine)

    coarse_f = coarse.float() / 127.5 - 1.
    fine_f = fine.float() / 127.5 - 1.

    return mels, coarse, fine, coarse_f, fine_f


def collate_emo_samples(left_pad, window, right_pad, batch):
    samples = [x[0] for x in batch]
    f0 = [x[1] for x in batch]
    global_cond = [x[2] for x in batch]
    #print(f0[0].shape, 111)

    max_offsets = [x.shape[-1] - window for x in samples]
    print(max_offsets, 'mo')
    f0_offsets = [np.random.randint(0, float(offset/config.hop_length)) for offset in max_offsets]
    sig_offsets = [np.random.randint(0, offset) for offset in max_offsets]
    print(f0_offsets, 'f0 off')
    f0 = [x[f0_offsets[i]:f0_offsets[i] + int(window/config.hop_length)] for i, x in enumerate(f0)]
    # print([len(x) for x in f0], len(f0))
    f0 = np.stack(f0).astype(np.float32)
    print(samples[0].shape, 'collate samples')
    print(window, left_pad, right_pad)
    wave16 = [np.concatenate([np.zeros(left_pad, dtype=np.int16), x, np.zeros(right_pad, dtype=np.int16)])[sig_offsets[i]:sig_offsets[i] + left_pad + window + right_pad] for i, x in enumerate(samples)]

    wave16 = torch.LongTensor(np.stack(wave16).astype(np.int64))
    f0 = torch.FloatTensor(f0)
    global_cond = torch.FloatTensor(global_cond)

    print(wave16.shape, f0.shape, 'collate f0')

    return wave16, f0, global_cond

def restore(path, model):
    model.load_state_dict(torch.load(path))

    match = re.search(r'_([0-9]+)\.pyt', path)
    if match:
        return int(match.group(1))

    step_path = re.sub(r'\.pyt', '_step.npy', path)
    return np.load(step_path)

if __name__ == '__main__':
    import pickle
    from torch.utils.data import DataLoader
    DATA_PATH = '/home/smg/zhaoyi/projects/emotion_enhancement/data/preprocessed_data'
    with open(f'{DATA_PATH}/index.pkl', 'rb') as f:
        index = pickle.load(f)
    gender_emo_path = config.gender_emo_path
    spk_emd_path = config.spk_emd_path
        #def __init__(self, train_index, path, gender_emo_path, spk_emd_path):
    dataset = EmospeakerDataset(index, DATA_PATH, gender_emo_path, spk_emd_path)
    loader = DataLoader(dataset, collate_fn=lambda batch: collate_emo_samples(0, 16, 0, batch), batch_size=2,
                            num_workers=2, shuffle=True, pin_memory=True)
    for x in loader:
        pass
