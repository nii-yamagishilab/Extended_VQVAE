# ==============================================================================
# Copyright (c) 2020, Yamagishi Laboratory, National Institute of Informatics
# Author: Yi Zhao (zhaoyi@nii.ac.jp)
# All rights reserved.
# ==============================================================================
#
#
import sys
import glob
import pickle
import os
import multiprocessing as mp
from utils.dsp import *
import pandas as pd
import multiprocessing as mp
import config

def read_binfile(filename, dim=60, dtype=np.float64):
    '''
    Reads binary file into numpy array.
    '''
    fid = open(filename, 'rb')
    v_data = np.fromfile(fid, dtype=dtype)
    fid.close()
    if np.mod(v_data.size, dim) != 0:
        raise ValueError('Dimension provided not compatible with file size.')
    m_data = v_data.reshape((-1, dim)).astype('float64') # This is to keep compatibility with numpy default dtype.
    m_data = np.squeeze(m_data)
    return  m_data

def process_file(wav_file, f0_file, out_path):
    name = wav_file.split('/')[-1][:-4] # Drop .wav

    filename = f'{out_path}/quant/{name}.npy'
    # if os.path.exists(filename):
    #     print(f'{filename} already exists, skipping')
    #     return

    floats = load_wav(wav_file, encode=False)

    # trimmed, _ = librosa.effects.trim(floats, top_db=25) ##trimmed before
    quant = (floats * (2**15 - 0.5) - 0.5).astype(np.int16)
    if max(abs(quant)) < 2048:
        print(f'audio fragment too quiet ({max(abs(quant))}), skipping: {wav_file}')
        return
    if len(quant) < 5000:
        print(f'audio fragment too short ({len(quant)} samples), skipping: {wav_file}')
        return
    os.makedirs(out_path, exist_ok=True)


    f0 = read_binfile(f0_file, dim=1)
    filename_f0 = f'{out_path}/f0/{name}.npy'


    audio_ref_size = len(f0)*config.hop_length

    frm_diff = len(quant) - audio_ref_size
    if frm_diff <0:
        quant = np.r_[ quant, np.zeros(-frm_diff) + quant[-1]]
    if frm_diff > 0:
        quant = quant[:-frm_diff]

    if  abs(len(quant)-len(f0)*config.hop_length) > 2 :
        print(len(quant), len(f0), len(quant)-len(f0)*config.hop_length, filename)

    np.save(filename, quant)
    np.save(filename_f0, f0)
    return name


def main():
    data_path = sys.argv[1]
    wav_path = os.path.join(data_path, 'wavs', 'wav_22050_trimmed_normalized')
    if0_path = os.path.join(data_path, 'features', 'qf0_22050_64')
    scp_path = os.path.join(data_path, 'scp')
    all_scp = os.path.join(data_path, 'all.scp')
    gender_emo_scp = os.path.join(scp_path, 'utt_gender_emotion.csv')
    spk_emd_scp = os.path.join(scp_path, 'spk_ebd.csv')
    out_path = os.path.join(data_path, 'preprocessed_data')

    df_gender_emo = pd.read_csv(gender_emo_scp, sep="\t", dtype=object, header=0)
    df_spk_emd = pd.read_csv(spk_emd_scp, sep="\t", dtype=object, header=0)


    files = [os.path.join(wav_path, x)+'.wav' for x in df_gender_emo['utt'].values]

    os.makedirs(os.path.join(out_path, 'quant'), exist_ok=True)
    os.makedirs(os.path.join(out_path, 'f0'), exist_ok=True)
    index = []
    with mp.Pool(2*mp.cpu_count()+1) as pool:
        res = pool.starmap_async(process_file, [(f, os.path.join(if0_path, os.path.basename(f).split('.')[0]+'.qf0'), out_path) for f in files]).get()

        index.extend([x for x in res if x])


    # index = df_gender_emo['utt'].values
    with open(f'{out_path}/index.pkl', 'wb') as f:
        pickle.dump(index, f)





if __name__ == '__main__':
    main()
