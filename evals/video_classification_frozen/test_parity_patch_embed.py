# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Quick script to show the disparity beween buffer-based 
# Sinusoidal3dPositionEncoder and the parameter-based variant

import os

from fairseq2.assets import default_asset_store, setup_asset_store
from fairseq2.logging import get_log_writer
from fairseq2.models.jepa import load_jepa_model
from fairseq2.recipes.utils.setup import setup_root_gang
from fairseq2.typing import CPU
import torch

from evals.video_classification_frozen.eval_fs2 import init_model

setup_asset_store(default_asset_store)
log = get_log_writer(__name__)


def main(args, resume_preempt=False):
    # Copy the params setup from evals.video_classification_frozen.eval_fs2.main()
    
    # -- PRETRAIN
    args_pretrain = args.get("pretrain")
    model_name = args_pretrain.get("model_name")
    checkpoint_key = args_pretrain.get('checkpoint_key', 'target_encoder')
    model_name = args_pretrain.get('model_name', None)
    patch_size = args_pretrain.get('patch_size', None)
    pretrain_folder = args_pretrain.get('folder', None)
    ckp_fname = args_pretrain.get('checkpoint', None)
    use_sdpa = args_pretrain.get('use_sdpa', True)
    use_SiLU = args_pretrain.get('use_silu', False)
    tight_SiLU = args_pretrain.get('tight_silu', True)
    uniform_power = args_pretrain.get('uniform_power', False)
    pretrained_path = os.path.join(pretrain_folder, ckp_fname)
    # Optional [for Video model]:
    tubelet_size = args_pretrain.get('tubelet_size', 2)
    pretrain_frames_per_clip = args_pretrain.get('frames_per_clip', 1)

    # -- DATA
    args_data = args.get('data')
    resolution = args_data.get('resolution', 224)
    
    # ----------------------------------------------------------------------- #

    gang = setup_root_gang(log)
    
    # Initialize fs2 JEPA model
    jepa_model = load_jepa_model(model_name, device=CPU, dtype=torch.float32)
    
    # Initialize original model
    encoder = init_model(
        crop_size=resolution,
        device=gang.device,
        pretrained=pretrained_path,
        model_name="vit_large",
        patch_size=patch_size,
        tubelet_size=tubelet_size,
        frames_per_clip=pretrain_frames_per_clip,
        uniform_power=uniform_power,
        checkpoint_key=checkpoint_key,
        use_SiLU=use_SiLU,
        tight_SiLU=tight_SiLU,
        use_sdpa=use_sdpa)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    
    # frequency tables in JEPA are 2D, while the original one is flatten to 1D    
    freqs = jepa_model.encoder_frontend.pos_encoder.freqs
    freqs = freqs.unsqueeze(0).flatten(1, -2)  # (H_frq, W_frq, E, D) -> (1, H_frq, W_frq, E, D) -> (1, N, D)
    
    print(f"Frequency table as loaded from the checkpoint: {encoder.pos_embed}")
    print(f"Frequency table as computed: {freqs}")
    
    torch.testing.assert_close(freqs, encoder.pos_embed)


if __name__ == "__main__":
    from fire import Fire
    Fire()