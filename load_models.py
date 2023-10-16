import os, sys

current_file_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_file_directory)

import legacy
from training.networks_stylegan3_editing import Generator

def load_stylegan(img_resolution, new_model=True, cfg = 'stylegan3-r', unaligned=False):
    model_root_dir = f'/engram/nklab/face_proj/models/{cfg}'
    assert os.path.exists(model_root_dir), f'No model directory found at {model_root_dir}'
    if img_resolution == 128 and unaligned is False:
        network_pkl = "ffhq_128/00000-stylegan3-r-ffhq_128-gpus8-batch32-gamma0.5/network-snapshot-025000.pkl"
    elif img_resolution == 256:
        network_pkl = "stylegan3-r-ffhqu-256x256.pkl"
    elif img_resolution == 1024 and unaligned is False:
        network_pkl = "stylegan3-r-ffhq-1024x1024.pkl"
    else:
        raise NotImplementedError(f'No model found for resolution {img_resolution} trained on unaligned={unaligned} dataset')
    
    network_pkl = os.path.join(model_root_dir, network_pkl)
    with open(network_pkl, 'rb') as f:
        G = legacy.load_network_pkl(f)['G_ema'].to('cuda')
    
    if new_model is False:
        return G
    
    channel_base = 16384 if img_resolution <= 256 else 32768
    print(channel_base)
    channel_max = 512
    mapping_kwargs = {"num_layers": 2}
    conv_kernel = 3
    use_radial_filters = False
    
    if cfg == 'stylegan3-r':
        conv_kernel = 1 # Use 1x1 convolutions.
        channel_base *= 2 # Double the number of feature maps.
        channel_max *= 2
        use_radial_filters = True # Use radially symmetric downsampling filters.
    else:
        raise NotImplementedError
    
    new_G = Generator(
        z_dim=512,
        c_dim=0,
        w_dim=512,
        img_resolution=img_resolution,
        img_channels=3,
        channel_base=channel_base,
        channel_max=channel_max,
        conv_kernel=conv_kernel,
        use_radial_filters=use_radial_filters,
        mapping_kwargs=mapping_kwargs
    )
    new_G.load_state_dict(G.state_dict())
    return new_G