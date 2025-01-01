# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Utility functions for fairseq2-ported checkpoints

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

import torch
from torch import Tensor
import torch.nn as nn

from fairseq2.assets import AssetCard, InProcAssetMetadataProvider, default_asset_store, setup_asset_store
from fairseq2.models.jepa.classifier import JepaClassifierModel
from fairseq2.models.jepa import load_jepa_config

from src.models.attentive_pooler import AttentiveClassifier
from src.models.vision_transformer import VisionTransformer

setup_asset_store(default_asset_store)


def create_model_card(checkpoint: Path, model_arch: str, model_family: str, pretrain_model_card: str, **kwargs) -> AssetCard:
    """Create a model card on-the-fly for fine-tuned v-jepa model"""

    def patch_tuples(node, *keys):
        for key in keys:
            if isinstance(node[key], tuple):
                node[key] = list(node[key])

    pretrained_config = load_jepa_config(pretrain_model_card)
    name = "on_the_fly_vjepa"
    model_card_info = {
        "name": name,
        "model_family": model_family,
        "model_arch": model_arch,
        "checkpoint": "file://" + checkpoint.as_posix(),
    }

    model_card_info["model_config"] = deepcopy(kwargs)
    model_card_info["model_config"]["encoder_config"] = asdict(pretrained_config.encoder_config)

    # Patch tuples since config_loader accepts only list
    patch_tuples(
        model_card_info["model_config"]["encoder_config"],
        "input_dims", "patch_dims",
    )

    metadata_provider = InProcAssetMetadataProvider([model_card_info])

    default_asset_store.user_metadata_providers.append(metadata_provider)

    try:
        return default_asset_store.retrieve_card(name)
    finally:
        default_asset_store.user_metadata_providers.pop()


class Aggregator(nn.Module):
    def __init__(
        self,
        model: JepaClassifierModel,
        encoder: VisionTransformer,
        classifier: AttentiveClassifier,
        tubelet_size: int = 2,
        attend_across_segments: bool = False,
    ):
        super().__init__()
        self.model = model
        self.encoder = encoder
        self.classifier = classifier
        self.tubelet_size = tubelet_size
        self.attend_across_segments = attend_across_segments

    def forward(self, clips: List[List[Tensor]], clip_indices: Optional[List[Tensor]] = None) -> List[Tensor]:
        num_clips = len(clips)
        num_views_per_clip = len(clips[0])
        B, C, T, H, W = clips[0][0].size()

        # Concatenate all spatial and temporal views along batch dimension
        x = [torch.cat(clip, dim=0) for clip in clips]
        x = torch.cat(x, dim=0)
        
        # inject checkpointed position encoder into fs2 model
        freqs_shape = self.model.encoder_frontend.pos_encoder.freqs.shape
        self.model.encoder_frontend.pos_encoder.freqs = self.model.encoder.pos_embed.squeeze().reshape(freqs_shape)

        features = self.model.encoder_frontend(seqs=x, padding_mask=None)
        embed, _ = self.model.encoder(*features)  # [batch x num_views_per_clip x num_clips, num_tokens, embed_dim]
        
        output_orig = self.encoder(x)
        torch.testing.assert_close(embed, output_orig, atol=1e-5, rtol=1e-5)

        _, num_tokens, D = embed.size()
        T = T // self.tubelet_size  # Num temporal tokens
        N = num_tokens // T  # Num spatial tokens

        # Unroll outputs into a 2D array [spatial_views x temporal_views]
        for i in range(num_clips):
            o = embed[i*eff_B:(i+1)*eff_B]
            for j in range(num_views_per_clip):
                view_embeds[j].append(o[j*B:(j+1)*B])
                
        view_outputs = []
        for i, view_embed in enumerate(view_embeds):

            # Concatenate along temporal dimension
            view_embed = [o.reshape(B, T, N, D) for o in view_embed]
            view_embed = torch.cat(view_embed, dim=1).flatten(1, 2)  # [batch, num_tokens_in_a_single_view, embed_dim]
            
            view_pool = self.model.pooler(view_embed)
            view_pool = view_pool.squeeze(1)  # Remove temporal dimension as all are attended into one pooled vector
            view_output = self.model.head(view_pool) 
            
            # Check parity with the classifier
            view_output_orig = self.classifier(view_output)
            breakpoint()            
            torch.testing.assert_close(view_output, view_output_orig, atol=1e-5, rtol=1e-5)
            
            view_outputs.append(view_output)

        return view_outputs
