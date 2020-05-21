# ==============================================================================
# Copyright (c) 2020, Yamagishi Laboratory, National Institute of Informatics
# Author: Yi Zhao (zhaoyi@nii.ac.jp)
# All rights reserved.
# ==============================================================================
import torch
import torch.nn.functional as F

def sample_softmax(score):
    """ Sample from the softmax distribution represented by scores.

    Input:
        score: (N, D) numeric tensor
    Output:
        sample: (N) long tensor, 0 <= sample < D
    """

    # Softmax tends to overflow fp16, so we use fp32 here.
    posterior = F.softmax(score.float(), dim=1)
    return torch.distributions.Categorical(posterior).sample()
