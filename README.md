# Extended_VQVAE
This is a Pytorch implementation of extended VQVAE mentioned in [our paper](https://arxiv.org/abs/2005.07884).

We introduce an important extension to VQ-VAE for learning F0-related suprasegmental information simultaneously along with traditional phone features. The proposed framework uses two encoders such that the F0 trajectory and speech waveform are both input to the system, therefore two separate codebooks are learned. 

![Framework of extended VQVAE](https://github.com/nii-yamagishilab/Extended_VQVAE/blob/master/framework.png?raw=true)

We reconstructed the speech using both [original VQVAE](https://arxiv.org/abs/1711.00937) and extended VQVAE with [F0 encoder](https://arxiv.org/abs/2005.07884). 

In brief, we have done:

1. Extended the original VQ-VAE with an F0 encoder.
2. Extended the global condition to speaker code, gender code, and emotion code. 
3. Trained a model with multi-speaker & multi-emotional Japanese corpus.
4. Trained a model with a public Chinese corpus.
5. A parallel training script on multiple gpus.

To do:
1. Testing on voice conversion
2. Testing on emotions' conversion

# Authors 
Authors of the paper: Yi Zhao,  Haoyu Li,  Cheng-I Lai, Jennifer Williams, Erica Cooper, Junichi Yamagishi

For any question related to the paper or the scripts, please contact zhaoyi[email mark]nii.ac.jp.

# Samples
Please find our samples [here](https://nii-yamagishilab.github.io/yi-demo/interspeech-2020/index.html).

# Requirements

Please install packages in requirement.txt before using the scripts.

# Preprocessing
1. extract F0. (We used [crepe](https://github.com/marl/crepe) to extract F0.  )
2. F0 and Wavefrom Alignment
3. converting F0 and waveform into *.npy format.


# Usage
Please use ./run.sh  when train an extended vavae model.

Or you can use python3 train.py -m [model type]. The -m option can be used to tell the the script to train a different model.

[model type] can be:
- 'vqvae': Train original VQVAE
- 'wavernn': train an WaveRNN model
- 'vcf0': extended VQVAE with F0 encoder

Please modify sampling rate and other parameters in [config.py](https://github.com/nii-yamagishilab/Extended_VQVAE/blob/master/config.py) before training.


# Trained models
We have Japanese or Chinese trained models for both original VQVAE and extended VQVAE. If you’re interested in using our pre-trained models for research purpose, please contact the zhaoyi[email mark]nii.ac.jp.

# Multi-gpu parallel training
Please see [multi_gpu_wavernn.py](https://github.com/nii-yamagishilab/Extended_VQVAE/blob/master/multi_gpu_wavernn.py)

# Acknowledgement

The code is based on [mkotha/WaveRNN](https://github.com/mkotha/WaveRNN)

Cheng-I is supported by the Merrill Lynch Fel- lowship, MIT. This work was partially supported by a JST CREST Grant (JPMJCR18A6, VoicePersonae project), Japan, and by MEXT KAKENHI Grants (16H06302, 18H04112, 18KT0051, 19K24373
Japan. The numerical calculations were carried out on the TSUBAME 3.0 supercomputer at the Tokyo Institute of Technology.

# License

MIT License
- Copyright (c) 2019 fatchord (https://github.com/fatchord)
- Copyright (c) 2019 mkotha (https://github.com/mkotha)
- Copyright (c) 2020, Yamagishi Laboratory, National Institute of Informatics.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
