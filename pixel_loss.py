import os, sys

sys.path.insert(1, os.path.join(sys.path[0], ".."))
import re
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn

from stylegan3.load_models import load_stylegan
from stimulus_optimization.generator_objects import ModelParallel

argparser = argparse.ArgumentParser()
argparser.add_argument("--batch_size", type=int, required=True, default=16)
argparser.add_argument(
    "--img_resolution", type=int, required=True, default=128, choices=[128, 256, 1024]
)
argparser.add_argument("--aligned", default=False, action="store_true")
args = argparser.parse_args()

batch_size = args.batch_size
model_device = torch.device("cuda:1")
generator_device = [torch.device("cuda:{}".format(i)) for i in range(2)]
param_device = generator_device[0]

G = load_stylegan(
    img_resolution=args.img_resolution,
    new_model=True,
    cfg="stylegan3-r",
    aligned=args.aligned,
)

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
    w_target = G_mapping(z_target, c_target, truncation_psi=0.5, truncation_cutoff=None)
    target_ims = G_synthesis(w_target, noise_mode="const", force_fp32=True)

optimized_types = ["z", "w", "s"]

all_output = []
all_loss = []

for optimized in optimized_types:
    for i_batch in range(1):
        z_param = torch.randn([batch_size, z_dim]).to(param_device)

        if optimized == "w":
            w_avg = G_mapping.model_replicas[0].w_avg
            w_init = torch.tile(w_avg, (batch_size, 1))
            w_init.requires_grad_(True)
            optimized_params = {optimized: w_init}
        elif optimized == "z":
            z_param = torch.randn([batch_size, z_dim]).to(param_device)
            z_param.requires_grad_(True)
            optimized_params = {optimized: z_param}
        elif optimized == "s":
            w_avg = G_mapping.model_replicas[0].w_avg
            w_avg = torch.tile(w_avg, (batch_size, 16, 1))
            s_init = G_synthesis.model_replicas[0].W2S(w_avg)
            optimized_params = {}
            for key, value in s_init.items():
                optimized_params[key] = value.detach().clone()
                optimized_params[key].requires_grad_(True)

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
            for i in (pbar := tqdm(range(500))):
                optimizer.zero_grad()
                if optimized == "s":
                    output_ims = G_synthesis(
                        optimized_params, noise_mode="const", force_fp32=True
                    ).to(param_device)
                else:
                    if optimized == "z":
                        w_param = G_mapping(
                            z_param, None, truncation_psi=0.5, truncation_cutoff=None
                        )
                    elif optimized == "w":
                        w_param = w_init.unsqueeze(1).repeat([1, G.num_ws, 1])
                    output_ims = G_synthesis(
                        w_param, noise_mode="const", force_fp32=True
                    ).to(param_device)

                loss = torch.mean((target_ims.flatten(1) - output_ims.flatten(1)) ** 2)
                pbar.set_description("loss: %0.4f" % (loss))

                loss.backward()
                optimizer.step()
                if optimized == "z":
                    optimized_params["z"] = optimized_params["z"] / torch.linalg.norm(
                        optimized_params["z"], dim=1
                    ).reshape(-1, 1)

        with torch.no_grad():
            all_output.append(output_ims)
            all_loss.append(
                torch.mean((target_ims.flatten(1) - output_ims.flatten(1)) ** 2, dim=1)
                .cpu()
                .numpy()
            )
