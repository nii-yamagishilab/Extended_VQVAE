# Extended_VQVAE
This is a Pytorch implementation of extended VQVAE mentioned in [our paper](https://arxiv.org/abs/2005.07884).

This paper introduces an important extension to VQ-VAE for learning F0-related suprasegmental infor- mation simultaneously along with traditional phone features. The proposed framework uses two encoders such that the F0 trajectory and speech waveform are both input to the system, there- fore two separate codebooks are learned.

![Framework of extended VQVAE](https://github.com/nii-yamagishilab/Extended_VQVAE/blob/master/framework.png?raw=true)

We reconstructed the speech using both [original VQVAE](https://arxiv.org/abs/1711.00937) and extended VQVAE with [F0 encoder](https://arxiv.org/abs/2005.07884). 


# Authors 
Authors of the paper: Yi Zhao,  Haoyu Li,  Cheng-I Lai, Jennifer Williams, Erica Cooper, Junichi Yamagishi

For any question related to the paper or the scripts, please contact zhaoyi[email mark]nii.ac.jp.

# Samples
Please find our samples [here](https://nii-yamagishilab.github.io/yi-demo/interspeech-2020/index.html).

# Usage
./run.sh    
(will add more information here)

# Trained models
We can provide trained models for only research purpose.  We have trained models for both original VQVAE and extended VQVAE. Please contact the zhaoyi[email mark]nii.ac.jp if you want to get either Japanese or Chinese trained models. 
;1. Japanese 
;- Extended VQVAE:[checkpoints/jp_vcf0.43.upconv.pyt](https://github.com/nii-yamagishilab/VC_VQVAE/blob/master/checkpoints/jp_vcf0.43.upconv.pyt)
;- Original VQVAE:[checkpoints/jp_vqvae.43.upconv.pyt](https://github.com/nii-;yamagishilab/VC_VQVAE/blob/master/checkpoints/jp_vqvae.43.upconv.pyt)
           
;2. Chinese
;- Extended VQVAE:[checkpoints/ch_vcf0.43.upconv.pyt](https://github.com/nii-;yamagishilab/VC_VQVAE/blob/master/checkpoints/ch_vcf0.43.upconv.pyt) 
; - Original VQVAE: [checkpoints/ch_vqvae.43.upconv.pyt](https://github.com/nii-;yamagishilab/VC_VQVAE/blob/master/checkpoints/ch_vqvae.43.upconv.pyt)       
     

# Acknowledgement

The code is based on [mkotha/WaveRNN](https://github.com/mkotha/WaveRNN)

# License

MIT License

Copyright (c) 2019 mkotha (https://github.com/mkotha)
Copyright (c) 2020 YiAthena (https://github.com/YiAthena)
Copyright (c) 2020, Yamagishi Laboratory, National Institute of Informatics.

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
