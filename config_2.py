# ==============================================================================
# Copyright (c) 2020, Yamagishi Laboratory, National Institute of Informatics
# Original: https://github.com/mkotha/WaveRNN
# Modified: Yi Zhao (zhaoyi[at]nii.ac.jp)
# All rights reserved.
# ==============================================================================
#
#
#single_speaker_data_path = '/home/smg/zhaoyi/projects/neural_vocoder/wavernn_test/data/vctk_neural_vocoder/hop256'
single_speaker_data_path = '/home/smg/zhaoyi/projects/neural_vocoder/wavernn_test/data/vctk_neural_vocoder/waveglow'
sample_rate = 22050
n_fft = 2048
fft_bins = n_fft // 2 + 1
num_mels = 80
hop_length = 256
win_length = 1024
fmin = 0
fmax = 8000
min_level_db = -100
ref_level_db = 20
upsample_factors = (8, 8, 4)
checkpoint_dir = "vctk_checkpoints_2"
output_dir="vctk_outputs_2"
rnn_dims=1024
fc_dims=1024
