import os, sys
sys.path.insert(1, os.path.join(sys.path[0], ".."))
import re
import argparse

import torch
import torch.nn as nn

from stylegan3 import legacy

from model_zoo import VGG16_BFM_128
from stimulus_optimization.activation import get_layerwise_activations

parser=argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, required=True)
args=parser.parse_args()

model_device=torch.device('cuda:2')
generator_device = [torch.device('cuda:{}'.format(i)) for i in range(2)]
param_device=generator_device[0]
generator_device_idx=[device.index for device in generator_device]

network_pkl = "/engram/nklab/face_proj/models/stylegan3/ffhq_128/00000-stylegan3-r-ffhq_128-gpus8-batch32-gamma0.5/network-snapshot-025000.pkl"
with open(network_pkl, 'rb') as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(param_device)
z_dim = G.z_dim
G= nn.DataParallel(G, device_ids=generator_device_idx)
import pdb; pdb.set_trace()

def normalized_sq_err(act_cur, act_target, act_target_norm):
    feature_diff = (act_cur.flatten(1) - act_target.flatten(1))
    feature_diff_norm = torch.norm(feature_diff, p=2, dim=-1)
    normalized_sq_err = feature_diff_norm / act_target_norm
    return torch.mean(normalized_sq_err)

with torch.no_grad():
    batch_size = args.batch_size
    z_target = torch.randn([batch_size, z_dim]).to(param_device)    # latent codes
    c_target = None                                # class labels (not used in this example)
    # z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
    target_ims = G(z_target, c_target, truncation_psi=0.5, truncation_cutoff=None, noise_mode='const', force_fp32=True)

    face_model=VGG16_BFM_128(0)
    face_model.load(device=model_device)
    layer_idx=34

    act_target=get_layerwise_activations(
        model=face_model, 
        layer_indices=[layer_idx], 
        ims=target_ims, 
        compression=None,
        forward=True, 
        in_place=False)[0].to(param_device)
    act_target.requires_grad_(False)
    act_target_norm = torch.norm(act_target.flatten(1), p=2, dim=-1)
    
z_param = torch.randn([batch_size, z_dim]).to(param_device)
z_param.requires_grad_(True)
lr = 0.01  # the learning rate
betas = (0.9, 0.999)  # the coefficients used for computing running averages of gradient and its square
eps = 1e-8
optimizer = torch.optim.Adam([z_param], lr=lr) #, betas=betas, eps=eps
lambda_reg = 0
with torch.autograd.detect_anomaly():
    for i in range(100):
        optimizer.zero_grad()
        output_ims = G(z_param, [None]*batch_size, truncation_psi=0.5, truncation_cutoff=None, noise_mode='const', force_fp32=True)
        act_cur=get_layerwise_activations(model=face_model, layer_indices=[layer_idx], ims=output_ims, compression=None, forward=True, in_place=False)[0]
        loss = normalized_sq_err(act_cur.to(param_device), act_target, act_target_norm)

        loss.backward()
        if i%10==0:
            print("loss")
            print(loss.item())
        optimizer.step()
