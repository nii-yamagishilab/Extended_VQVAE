# ==============================================================================
# Copyright (c) 2020, Yamagishi Laboratory, National Institute of Informatics
# Original: https://github.com/mkotha/WaveRNN
# Modified: Yi Zhao (zhaoyi[at]nii.ac.jp)
# All rights reserved.
# ==============================================================================
#
#
#!/bin/sh

#python preprocess_f0.py /home/smg/zhaoyi/projects/emotion_enhancement/data

python3 train.py -m vcf0 -c 1
