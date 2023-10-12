import os, sys

sys.path.insert(1, os.path.join(sys.path[0], ".."))
import argparse
from tqdm import tqdm

import numpy as np
from numpy import arccos
from numpy.linalg import norm
import torch
import torch.nn as nn

from stylegan3 import legacy

from model_zoo import VGG16_BFM_128
from stimulus_optimization.activation import get_layerwise_activations
from stimulus_optimization.generator_objects import ModelParallel

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, required=True)
parser.add_argument("--optimized", type=str, required=True)
parser.add_argument("--truncation_psi", type=float, default=0.5)
parser.add_argument("--num_seeds", type=int, default=5)
args = parser.parse_args()

batch_size = args.batch_size
optimized = args.optimized
truncation_psi = args.truncation_psi


def theta(v, w):
    dot = (v @ w.T).diagonal()
    return arccos(dot / (norm(v, axis=1) * norm(w, axis=1)))


def normalized_sq_err(act_cur, act_target, act_target_norm):
    feature_diff = act_cur.flatten(1) - act_target.flatten(1)
    feature_diff_norm = torch.norm(feature_diff, p=2, dim=-1)
    normalized_sq_err = feature_diff_norm / act_target_norm
    return torch.mean(normalized_sq_err)


def postprocess_torch(ims):
    ims = ims.clone()
    return (ims * 127.5 + 128).clamp(0, 255).to(torch.uint8).detach()


model_device = torch.device("cuda:1")
generator_device = [torch.device("cuda:{}".format(i)) for i in range(2)]
param_device = generator_device[0]

network_pkl = "/engram/nklab/face_proj/models/stylegan3/ffhq_128/00000-stylegan3-r-ffhq_128-gpus8-batch32-gamma0.5/network-snapshot-025000.pkl"
with open(network_pkl, "rb") as f:
    G = legacy.load_network_pkl(f)["G_ema"].to(param_device)

z_dim = G.z_dim
G_mapping = ModelParallel(
    G.mapping, devices=generator_device, output_device=param_device
)
G_synthesis = ModelParallel(
    G.synthesis, devices=generator_device, output_device=param_device
)

with torch.no_grad():
    z_target = torch.randn([batch_size, z_dim]).to(param_device)
    c_target = None
    w_target = G_mapping(
        z_target, c_target, truncation_psi=truncation_psi, truncation_cutoff=None
    )
    target_ims = G_synthesis(w_target, noise_mode="const", force_fp32=True)

    face_model = VGG16_BFM_128(0)
    face_model.load(device=model_device)
    layer_idx = 34

    act_target = get_layerwise_activations(
        model=face_model,
        layer_indices=[layer_idx],
        ims=target_ims,
        compression=None,
        forward=True,
        in_place=False,
    )[0].to(param_device)
    act_target.requires_grad_(False)
    act_target_norm = torch.norm(act_target.flatten(1), p=2, dim=-1)

optimized = "w"
all_metamers = []
all_dissimilarities = []
all_angles = []

for i_batch in range(args.num_seeds):
    z_param = torch.randn([batch_size, z_dim]).to(param_device)

    if optimized == "w":
        w_init = G_mapping(z_param, None, truncation_psi=0.5, truncation_cutoff=None)
        w_init = w_init[:, 0, :]
        z_param.requires_grad_(False)
        w_init.requires_grad_(True)
        optimized_params = {optimized: w_init}
    elif optimized == "z":
        z_param.requires_grad_(True)
        optimized_params = {optimized: z_param}
    assert optimized_params[optimized].requires_grad

    lr = 0.01  # the learning rate
    betas = (
        0.9,
        0.999,
    )  # the coefficients used for computing running averages of gradient and its square
    eps = 1e-8
    optimizer = torch.optim.Adam(
        optimized_params.values(), lr=lr
    )  # , betas=betas, eps=eps
    with torch.autograd.detect_anomaly():
        for i in (pbar := tqdm(range(1000))):
            optimizer.zero_grad()
            if optimized == "z":
                w_param = G_mapping(
                    z_param, None, truncation_psi=0.5, truncation_cutoff=None
                )
            elif optimized == "w":
                w_param = w_init.unsqueeze(1).repeat([1, G.num_ws, 1])

            output_ims = G_synthesis(w_param, noise_mode="const", force_fp32=True)
            act_cur = get_layerwise_activations(
                model=face_model,
                layer_indices=[layer_idx],
                ims=output_ims,
                compression=None,
                forward=True,
                in_place=False,
            )[0]
            loss = normalized_sq_err(
                act_cur.to(param_device), act_target, act_target_norm
            )

            loss.backward()
            pbar.set_description(f"loss: {loss}")
            optimizer.step()
            if optimized == "z":
                optimized_params["z"] = optimized_params["z"] / torch.linalg.norm(
                    optimized_params["z"], dim=1
                ).reshape(-1, 1)

    with torch.no_grad():
        all_metamers.append(output_ims)
        new_dissimilarities = torch.sum(
            (act_cur.flatten(1).detach().cpu() - act_target.flatten(1).detach().cpu())
            ** 2,
            dim=1,
        )
        all_dissimilarities.append(new_dissimilarities.numpy())
        if optimized == "z":
            z_metamers = z_param.clone().detach().cpu().numpy()
            z_targets = z_target.clone().detach().cpu().numpy()
            angles = np.rad2deg(theta(z_metamers, z_targets))
            all_angles.append(angles)

metamers = torch.concat(all_metamers)
ims = torch.concat([target_ims, metamers])
ims = postprocess_torch(ims)
