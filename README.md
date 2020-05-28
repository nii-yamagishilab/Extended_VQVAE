# Extended_VQVAE
This is a Pytorch implementation of extended VQVAE mentioned in [our paper](https://arxiv.org/abs/2005.07884).This paper introduces an important extension to VQ-VAE for learning F0-related suprasegmental infor- mation simultaneously along with traditional phone features. The proposed framework uses two encoders such that the F0 trajectory and speech waveform are both input to the system, there- fore two separate codebooks are learned.

![Framework][./framework.png]

We reconstructed the speech using both [original VQVAE](https://arxiv.org/abs/1711.00937) and [F0 encoder](https://arxiv.org/abs/2005.07884). 

# Samples
Please find our samples [here](https://nii-yamagishilab.github.io/yi-demo/interspeech-2020/index.html).

# Usage
./run.sh    
(will add more information here)
# Trained models
1. Japanese 
- Extended VQVAE:[checkpoints/jp_vcf0.43.upconv.pyt](https://github.com/nii-yamagishilab/VC_VQVAE/blob/master/checkpoints/jp_vcf0.43.upconv.pyt)
- Original VQVAE:[checkpoints/jp_vqvae.43.upconv.pyt](https://github.com/nii-yamagishilab/VC_VQVAE/blob/master/checkpoints/jp_vqvae.43.upconv.pyt)
           
2. Chinese
- Extended VQVAE:[checkpoints/ch_vcf0.43.upconv.pyt](https://github.com/nii-yamagishilab/VC_VQVAE/blob/master/checkpoints/ch_vcf0.43.upconv.pyt) 
 - Original VQVAE: [checkpoints/ch_vqvae.43.upconv.pyt](https://github.com/nii-yamagishilab/VC_VQVAE/blob/master/checkpoints/ch_vqvae.43.upconv.pyt)       
     


# Acknowledgement

The code is based on [mkotha/WaveRNN](https://github.com/mkotha/WaveRNN)
