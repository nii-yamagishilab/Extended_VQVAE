# ==============================================================================
# Copyright (c) 2020, Yamagishi Laboratory, National Institute of Informatics
# Original: https://github.com/mkotha/WaveRNN
# Modified: Yi Zhao (zhaoyi[at]nii.ac.jp)
# All rights reserved.
# ==============================================================================
emo_data_path = '/home/smg/zhaoyi/projects/emotion_enhancement/data/preprocessed_data'
gender_emo_path = '/home/smg/zhaoyi/projects/emotion_enhancement/data/scp/utt_gender_emotion.csv'
spk_emd_path = '/home/smg/zhaoyi/projects/emotion_enhancement/data/scp/spk_ebd.csv'
sample_rate = 22050
n_fft = 2048
fft_bins = n_fft // 2 + 1
num_mels = 80
hop_length = 64
win_length = 1024
min_level_db = -100
ref_level_db = 20
fmin = 0
fmax = 8000
upsample_factors = (4, 4, 4)
checkpoint_dir = "nancy_mels-librosa_checkpoints"
output_dir="nancy_mels-librosa_output"
batch_size=50
rnn_dims=1024
fc_dims=1024
lr=1e-4
num_bit = 16

dim_speaker_embedding = 50
dim_gender_code = 10
dim_emotion_code = 40

test_files_txt ='/home/smg/zhaoyi/projects/neural_vocoder/data/nancy/mels-tacotron/mel-shinji/test.txt'
test_model_path = '/home/smg/zhaoyi/projects/neural_vocoder/wavernn_test/WaveRNN_VCTK_neural_vocoder/nancy_checkpoints2/wavernn.43.upconv.pyt'
test_output_dir = '/home/smg/zhaoyi/projects/neural_vocoder/data/nancy/samples_tacotron/samples_mel-shinji'
