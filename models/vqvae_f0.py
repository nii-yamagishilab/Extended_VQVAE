# ==============================================================================
# Copyright (c) 2020, Yamagishi Laboratory, National Institute of Informatics
# Original: https://github.com/mkotha/WaveRNN
# Modified: Yi Zhao (zhaoyi[at]nii.ac.jp)
# All rights reserved.
# ==============================================================================
import math, pickle, os
import numpy as np
import torch
from torch.autograd import Variable
from torch import optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from utils.dsp import *
import sys
import time
from layers.overtone import Overtone, Overtone_f0
from layers.vector_quant import VectorQuant
from layers.downsampling_encoder import DownsamplingEncoder
import utils.env as env
import utils.logger as logger
import random
from layers.upsample import UpsampleNetwork_F0
import pytorch_warmup as warmup
import config

class Model(nn.Module) :
    def __init__(self, rnn_dims, fc_dims, global_decoder_cond_dims, upsample_factors, normalize_vq=False,
            noise_x=False, noise_y=False):
        super().__init__()
        self.n_vq_classes = 512
        self.n_f0_classes = 128
        self.vec_len = 128
        # self.channel_f0 = 128
        #self.upsample = UpsampleNetwork_F0(upsample_factors)
        #n_channels, n_classes, vec_len, normalize=False

        self.vq = VectorQuant(1, self.n_vq_classes, self.vec_len, normalize=normalize_vq)
        self.vq_f0 = VectorQuant(1, self.n_f0_classes, self.vec_len, normalize=normalize_vq)
        #self.vq_f0 = VectorQuant(1, self.n_classes_f0, self.vec_len, normalize=normalize_vq)
        self.noise_x = noise_x
        self.noise_y = noise_y
        encoder_layers_wave = [
            (2, 4, 1),
            (2, 4, 1),
            (2, 4, 1),
            (1, 4, 1),
            (2, 4, 1),
            (1, 4, 1),
            (2, 4, 1),
            (1, 4, 1),
            (2, 4, 1),
            (1, 4, 1),
            ]
        self.encoder = DownsamplingEncoder(128, encoder_layers_wave)

        encoder_layers_f0 = [
            (2, 4, 1),
            (2, 4, 1),
            (2, 4, 1),
            (1, 4, 1),
            (2, 4, 1),
            (1, 4, 1),
            (2, 4, 1),
            (1, 4, 1),
            (2, 4, 1),
            (1, 4, 1),
            ]
        self.encoder_f0 = DownsamplingEncoder(128, encoder_layers_f0)
        self.frame_advantage = 15
        self.num_params()
        self.overtone = Overtone_f0(rnn_dims, fc_dims,self.vec_len*2, global_decoder_cond_dims)

    def forward(self, global_decoder_cond, x, samples, f0):  ##speaker , input_audio , noise+input_samples
        # x: (N, 768, 3)
        #logger.log(f'x: {x.size()}')
        # samples: (N, 1022)
        #logger.log(f'samples: {samples.size()}')
        continuous = self.encoder(samples)

        #f0_upsampled = self.upsample(f0)

        #continuous_f0 = self.encoder_f0 (f0_upsampled)
        continuous_f0 = self.encoder_f0 (f0)
        # continuous: (N, 14, 64)
        #logger.log(f'continuous: {continuous.size()}')
        discrete, vq_pen, encoder_pen, entropy = self.vq(continuous.unsqueeze(2))
        discrete_f0, vq_pen_f0, encoder_pen_f0, entropy_f0 = self.vq_f0(continuous_f0.unsqueeze(2))

        discrete = discrete.squeeze(2)
        code_x = discrete

        discrete_f0 = discrete_f0.squeeze(2)
        n_repeat = int(code_x.shape[1]/discrete_f0.shape[1])
        code_f0 = discrete_f0.repeat (1, n_repeat + 1, 1)[:,:discrete.shape[1],:]
        codes = torch.cat((code_x, code_f0) , dim=2 )

        return self.overtone(x, codes, global_decoder_cond), vq_pen.mean(), \
                encoder_pen.mean(), entropy, vq_pen_f0.mean(), encoder_pen_f0.mean(), entropy_f0

    def after_update(self):
        self.overtone.after_update()
        self.vq.after_update()

    def forward_generate(self, global_decoder_cond, samples, f0, deterministic=False, use_half=False, verbose=False):
        if use_half:
            samples = samples.half()
        # samples: (L)
        #logger.log(f'samples: {samples.size()}')
        self.eval()
        with torch.no_grad() :

            continuous = self.encoder(samples)

        #    f0_upsampled = self.upsample(f0)

        #    continuous_f0 = self.encoder_f0 (f0_upsampled)
            continuous_f0 = self.encoder_f0 (f0)
            # continuous: (N, 14, 64)
            #logger.log(f'continuous: {continuous.size()}')
            discrete, vq_pen, encoder_pen, entropy = self.vq(continuous.unsqueeze(2))
            discrete_f0, vq_pen_f0, encoder_pen_f0, entropy_f0 = self.vq_f0(continuous_f0.unsqueeze(2))

            discrete = discrete.squeeze(2)
            code_x = discrete

            discrete_f0 = discrete_f0.squeeze(2)
            n_repeat = int(code_x.shape[1]/discrete_f0.shape[1])
            code_f0 = discrete_f0.repeat (1, n_repeat + 1, 1)[:,:discrete.shape[1],:]
            codes = torch.cat((code_x, code_f0) , dim=2 )

            logger.log(f'entropy: {entropy}')
            output = self.overtone.generate(codes, global_decoder_cond, use_half=use_half, verbose=verbose)

        self.train()
        return output

    def num_params(self) :
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        logger.log('Trainable Parameters: %.3f million' % parameters)

    def load_state_dict(self, dict, strict=True):
        if strict:
            return super().load_state_dict(self.upgrade_state_dict(dict))
        else:
            my_dict = self.state_dict()
            new_dict = {}
            for key, val in dict.items():
                if key not in my_dict:
                    logger.log(f'Ignoring {key} because no such parameter exists')
                elif val.size() != my_dict[key].size():
                    logger.log(f'Ignoring {key} because of size mismatch')
                else:
                    logger.log(f'Loading {key}')
                    new_dict[key] = val
            return super().load_state_dict(new_dict, strict=False)

    def upgrade_state_dict(self, state_dict):
        out_dict = state_dict.copy()
        return out_dict


    def freeze_encoder(self):
        for name, param in self.named_parameters():
            if name.startswith('encoder.') or name.startswith('vq.'):
                logger.log(f'Freezing {name}')
                param.requires_grad = False
            else:
                logger.log(f'Not freezing {name}')

    def pad_left(self):
        return max(self.pad_left_decoder(), self.pad_left_encoder())

    def pad_left_decoder(self):
        return self.overtone.pad()

    def pad_left_encoder(self):
        return self.encoder.pad_left + (self.overtone.cond_pad - self.frame_advantage) * self.encoder.total_scale

    def pad_right(self):
        return self.frame_advantage * self.encoder.total_scale

    def total_scale(self):
        return self.encoder.total_scale

    def do_train(self, paths, dataset, optimiser, epochs, batch_size, step, lr=1e-3, valid_index=[], use_half=False, do_clip=False):

        if use_half:
            import apex
            optimiser = apex.fp16_utils.FP16_optimiser(optimiser, dynamic_loss_scale=True)
        for p in optimiser.param_groups : p['lr'] = lr
        criterion = nn.NLLLoss().cuda()
        k = 0
        saved_k = 0
        pad_left = self.pad_left()
        pad_left_encoder = self.pad_left_encoder()
        pad_left_decoder = self.pad_left_decoder()
        if self.noise_x:
            extra_pad_right = 127
        else:
            extra_pad_right = 0
        pad_right = self.pad_right() + extra_pad_right
        window = 16 * self.total_scale()
        logger.log(f'pad_left={pad_left_encoder}|{pad_left_decoder}, pad_right={pad_right}, total_scale={self.total_scale()}')

        #lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimiser, milestones=[num_epochs//3], gamma=0.1)
        lr_lambda = lambda epoch: min((epoch) / 100 , 1)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda=lr_lambda)
        #lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimiser, epochs=2000,
        #         steps_per_epoch=149, max_lr=0.001)
        #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, eta_min=0, T_max=2000)
    #    warmup_scheduler = warmup.UntunedLinearWarmup(optimiser)
    #    warmup_scheduler.last_step = -1



        for e in range(epochs) :
            trn_loader = DataLoader(dataset, collate_fn=lambda batch: env.collate_emo_samples(pad_left, window, pad_right, batch), batch_size=batch_size,
                                    num_workers=2, shuffle=True, pin_memory=True)

            start = time.time()
            running_loss_c = 0.
            running_loss_f = 0.
            running_loss_vq = 0.
            running_loss_vqc = 0.
            running_entropy = 0.

            running_loss_vq_f0 = 0.
            running_loss_vqc_f0 = 0.
            running_entropy_f0 = 0.

            running_max_grad = 0.
            running_max_grad_name = ""

            iters = len(trn_loader)

            for i, (wave16, f0, global_cond) in enumerate(trn_loader) :


                wave16 = wave16.cuda()
                f0 = f0.cuda()
                global_cond = global_cond.cuda()

                coarse = (wave16 + 2**15) // 256
                fine = (wave16 + 2**15) % 256

                coarse_f = coarse.float() / 127.5 - 1.
                fine_f = fine.float() / 127.5 - 1.
                total_f = (wave16.float() + 0.5) / 32767.5

                if self.noise_y:
                    noisy_f = total_f * (0.02 * torch.randn(total_f.size(0), 1).cuda()).exp() + 0.003 * torch.randn_like(total_f)
                else:
                    noisy_f = total_f

                if use_half:
                    coarse_f = coarse_f.half()
                    fine_f = fine_f.half()
                    noisy_f = noisy_f.half()

                x = torch.cat([
                    coarse_f[:, pad_left-pad_left_decoder:-pad_right].unsqueeze(-1),
                    fine_f[:, pad_left-pad_left_decoder:-pad_right].unsqueeze(-1),
                    coarse_f[:, pad_left-pad_left_decoder+1:1-pad_right].unsqueeze(-1),
                    ], dim=2)

                y_coarse = coarse[:, pad_left+1:1-pad_right]
                y_fine = fine[:, pad_left+1:1-pad_right]

                if self.noise_x:
                    # Randomly translate the input to the encoder to encourage
                    # translational invariance
                    total_len = coarse_f.size(1)
                    translated = []
                    for j in range(coarse_f.size(0)):
                        shift = random.randrange(256) - 128
                        translated.append(noisy_f[j, pad_left-pad_left_encoder+shift:total_len-extra_pad_right+shift])
                    translated = torch.stack(translated, dim=0)
                else:
                    translated = noisy_f[:, pad_left-pad_left_encoder:]
                    #global_decoder_cond, x, samples, f0
                #self.overtone(x, codes, global_decoder_cond), vq_pen.mean(), \
                        #encoder_pen.mean(), entropy, vq_pen_f0.mean(), encoder_pen_f0.mean(), entropy_f0
                p_cf, vq_pen, encoder_pen, entropy, vq_pen_f0, encoder_pen_f0, entropy_f0 = self(global_cond, x, translated, f0)
                p_c, p_f = p_cf
                loss_c = criterion(p_c.transpose(1, 2).float(), y_coarse)
                loss_f = criterion(p_f.transpose(1, 2).float(), y_fine)
                encoder_weight = 0.01 * min(1, max(0.1, step / 1000 - 1))
                # weight_f0 = e / 10.0
                weight_f0 = 100
                loss = loss_c + loss_f + vq_pen + encoder_weight * encoder_pen + (vq_pen_f0 + encoder_weight * encoder_pen_f0) * weight_f0

                optimiser.zero_grad()
                if use_half:
                    optimiser.backward(loss)
                    if do_clip:
                        raise RuntimeError("clipping in half precision is not implemented yet")
                else:
                    loss.backward()

                    if do_clip:
                        max_grad = 0
                        max_grad_name = ""
                        for name, param in self.named_parameters():
                            if param.grad is not None:
                                param_max_grad = param.grad.data.abs().max()
                                if param_max_grad > max_grad:
                                    max_grad = param_max_grad
                                    max_grad_name = name
                                if 1000000 < param_max_grad:
                                    logger.log(f'Very large gradient at {name}: {param_max_grad}')
                        if 100 < max_grad:
                            for param in self.parameters():
                                if param.grad is not None:
                                    if 1000000 < max_grad:
                                        param.grad.data.zero_()
                                    else:
                                        param.grad.data.mul_(100 / max_grad)
                        if running_max_grad < max_grad:
                            running_max_grad = max_grad
                            running_max_grad_name = max_grad_name

                        if 100000 < max_grad:
                            torch.save(self.state_dict(), "bad_model.pyt")
                            raise RuntimeError("Aborting due to crazy gradient (model saved to bad_model.pyt)")
                optimiser.step()
                lr_scheduler.step()
            #    warmup_scheduler.dampen()
                running_loss_c += loss_c.item()
                running_loss_f += loss_f.item()
                running_loss_vq += vq_pen.item()
                running_loss_vqc += encoder_pen.item()
                running_entropy += entropy

                running_loss_vq_f0 += vq_pen_f0.item()
                running_loss_vqc_f0 += encoder_pen_f0.item()
                running_entropy_f0 += entropy_f0

                self.after_update()

                speed = (i + 1) / (time.time() - start)
                avg_loss_c = running_loss_c / (i + 1)
                avg_loss_f = running_loss_f / (i + 1)
                avg_loss_vq = running_loss_vq / (i + 1)
                avg_loss_vqc = running_loss_vqc / (i + 1)
                avg_entropy = running_entropy / (i + 1)

                avg_loss_vq_f0 = running_loss_vq_f0 / (i + 1)
                avg_loss_vqc_f0 = running_loss_vqc_f0 / (i + 1)
                avg_entropy_f0 = running_entropy_f0 / (i + 1)

                step += 1
                k = step // 1000

            logger.status(f'Epoch:{e+1}/{epochs}--Batch:{i+1}/{iters}--Loss:c={avg_loss_c:#.4} f={avg_loss_f:#.4} vq={avg_loss_vq:#.4} vqc={avg_loss_vqc:#.4} vq_f0={avg_loss_vq_f0:#.4} vqc_f0={avg_loss_vqc_f0:#.4}  -- Entropy: {avg_entropy:#.4} -- Entropy_f0:{avg_entropy_f0:#.4}  -- Grad:{running_max_grad:#.1} {running_max_grad_name} Speed:{speed:#.4} steps/sec -- Step: {k}k ')
            os.makedirs(paths.checkpoint_dir, exist_ok=True)
            torch.save(self.state_dict(), paths.model_path())
            np.save(paths.step_path(), step)
            logger.log_current_status()
            logger.log(f' <saved>; w[0][0] = {self.overtone.wavernn.gru.weight_ih_l0[0][0]}')

            torch.save(self.state_dict(), paths.model_hist_path(step))
            if k > saved_k + 100:
                saved_k = k
                self.do_generate(paths, step, dataset.path, valid_index)

    def do_generate(self, paths, step, data_path, test_index, deterministic=False, use_half=False, verbose=False):
        k = step // 1000
        dataset = env.EmospeakerDataset(test_index, data_path, config.gender_emo_path, config.spk_emd_path)
        loader = DataLoader(dataset, shuffle=False)
        data = [x for x in loader]
        n_points = len(data)
        gt = [(x[0].float() + 0.5) / (2**15 - 0.5) for x, f0, global_cond in data]
        extended = [np.concatenate([np.zeros(self.pad_left_encoder(), dtype=np.float32), x, np.zeros(self.pad_right(), dtype=np.float32)]) for x in gt]

        f0s = [torch.FloatTensor(f0[0].float()) for x, f0, global_cond in data]
        global_conds = [torch.FloatTensor(global_cond[0].float()) for x, f0, global_cond in data]
        maxlen = max([len(x) for x in extended])
        aligned = [torch.cat([torch.FloatTensor(x), torch.zeros(maxlen-len(x))]) for x in extended]
        os.makedirs(paths.gen_path(), exist_ok=True)
        #global_decoder_cond, samples, f0,
        out = self.forward_generate(torch.stack(global_conds + list(reversed(global_conds)), dim=0).cuda(), torch.stack(aligned + aligned, dim=0).cuda(),  torch.stack(f0s + f0s, dim=0).cuda(),verbose=verbose, use_half=use_half)
        logger.log(f'out: {out.size()}')
        for i, x in enumerate(gt) :
            librosa.output.write_wav(f'{paths.gen_path()}/{k}k_steps_{i}_target.wav', x.cpu().numpy(), sr=sample_rate)
            audio = out[i][:len(x)].cpu().numpy()
            librosa.output.write_wav(f'{paths.gen_path()}/{k}k_steps_{i}_generated.wav', audio, sr=sample_rate)
            audio_tr = out[n_points+i][:len(x)].cpu().numpy()
            librosa.output.write_wav(f'{paths.gen_path()}/{k}k_steps_{i}_transferred.wav', audio_tr, sr=sample_rate)
