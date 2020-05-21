# ==============================================================================
# Copyright (c) 2020, Yamagishi Laboratory, National Institute of Informatics
# Author: Yi Zhao (zhaoyi@nii.ac.jp)
# All rights reserved.
# ==============================================================================
#emo_data_path = '/home/smg/zhaoyi/projects/emotion_enhancement/data/preprocessed_data_f0c'
#gender_emo_path = '/home/smg/zhaoyi/projects/emotion_enhancement/data/scp/utt_gender_emotion.csv'
#spk_emd_path = '/home/smg/zhaoyi/projects/emotion_enhancement/data/scp/spk_ebd.csv'
emo_data_path = '/home/smg/zhaoyi/projects/emotion_enhancement/data/corpus/BZNSYP/preprocessed_data_f0c'
gender_emo_path = '/home/smg/zhaoyi/projects/emotion_enhancement/data/corpus/BZNSYP/scp/BZNSYP_gender_emotion.csv'
spk_emd_path = '/home/smg/zhaoyi/projects/emotion_enhancement/data/corpus/BZNSYP/scp/BZNSYP_ebd.csv'
sample_rate = 22050
n_fft = 2048
fft_bins = n_fft // 2 + 1
num_mels = 80
# hop_length = 176
hop_length = 110
win_length = 1024 ## window / hop_size == 0
min_level_db = -100
ref_level_db = 20
fmin = 0
fmax = 8000
upsample_factors = (4, 4, 4, 4)
checkpoint_dir = "tmp2"
output_dir="tmp2"
# checkpoint_dir = "test_load_part"
# output_dir="out_load_part"
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

# test_files_txt ='test_file.txt'
# test_model_path = 'tmp/vcf0.43.upconv.pyt'
# test_output_dir = 'tmp_out_f20'

# test_files_txt = 'tmp.scp'
test_files_txt = 'test_file_ch.txt'
#test_files_txt ='/home/smg/zhaoyi/projects/emotion_enhancement/data/scp/system3_jvs.scp'
# test_model_path = 'checkpoints4/vcf0.43.upconv_86569.pyt'
#test_model_path = 'checkpoints_f06/vcf0.43.upconv_87165.pyt'
#'checkpoints_f06/vcf0.43.upconv.pyt'
#'checkpoints_f06_upconv_192806_system3'

# test_output_dir = 'system4'
# test_model_path = 'checkpoints_f20_f0c_3/vcf0.43.upconv_207808.pyt'
# test_output_dir = 'system3_34'
# test_model_path = 'checkpoints_ch/vcf0.43.upconv.pyt'
test_model_path = 'checkpoints_ch/vcf0.43.upconv.pyt'
test_output_dir = 'output_ch2'
