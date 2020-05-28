# ==============================================================================
# Copyright (c) 2020, Yamagishi Laboratory, National Institute of Informatics
# Author: Yi Zhao (zhaoyi@nii.ac.jp)
# All rights reserved.
# ==============================================================================

#Please change parameters


emo_data_path = 'data/preprocessed_data_f0c'
gender_emo_path = 'data/scp/utt_gender_emotion.csv'
spk_emd_path = 'data/scp/spk_ebd.csv'

sample_rate = 22050
n_fft = 2048
fft_bins = n_fft // 2 + 1
num_mels = 80
# hop_length = 176
#hop_length = 110
hop_length = 64
#
win_length = 1024
min_level_db = -100
ref_level_db = 20
fmin = 0
fmax = 8000
upsample_factors = (4, 4, 4) #4*4*4=64==hop_length
checkpoint_dir = "checkpoints"
output_dir="outputs"
test_load_part = True
batch_size=180
rnn_dims=1024
fc_dims=1024
lr=0.0001
num_bit = 16
num_epochs = 5000

dim_speaker_embedding = 50
dim_gender_code = 10
dim_emotion_code = 40



test_files_txt = 'data/test.scp'
test_model_path = 'checkpoints/vcf0.43.upconv.pyt'
test_output_dir = 'data/test_out'
