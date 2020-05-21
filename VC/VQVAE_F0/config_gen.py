# ==============================================================================
# Copyright (c) 2020, Yamagishi Laboratory, National Institute of Informatics
# Author: Yi Zhao (zhaoyi@nii.ac.jp)
# All rights reserved.
# ==============================================================================
#test_files_txt ='/home/smg/zhaoyi/projects/neural_vocoder/wavernn_test/data/vctk_neural_vocoder/hop256/vctk_test.txt'
#test_files_txt='vctk_copy_synthesis_text.txt'
#test_files_txt='vctk_tts_test.txt'
#test_files_txt = '/home/smg/zhaoyi/projects/neural_vocoder/wavernn_test/data/urmp_neural_vocoder/split/test_mel/test.txt'
# test_files_txt = 'urmp_test.txt'
# #output_dir = 'urmp_adapt'
# output_dir = 'urmp_adapt_gen'
# #output_dir = 'eval_vctk_copy_synthesis'
# #output_dir = 'urmp_zero_shot'
# #model_path = '/home/smg/zhaoyi/projects/neural_vocoder/wavernn_test/WaveRNN_VCTK_neural_vocoder/vctk_checkpoints_2/wavernn.43.upconv.pyt'
# model_path = '/home/smg/zhaoyi/projects/neural_vocoder/wavernn_test/WaveRNN_VCTK_neural_vocoder/urmp_checkpoints_adapt/wavernn.43.upconv.pyt'
# sample_rate = 22050
# batch_size = 10
# hop_length = 256

test_files_txt ='/home/smg/zhaoyi/projects/neural_vocoder/data/nancy/scp/test_nancy.txt'
model_path = '/home/smg/zhaoyi/projects/neural_vocoder/wavernn_test/WaveRNN_VCTK_neural_vocoder/nancy_checkpoints2/wavernn.43.upconv.pyt'
output_dir = '/home/smg/zhaoyi/projects/neural_vocoder/data/nancy/samples_mel-Shinji'
sample_rate = 24000
batch_size = 1
hop_length = 288