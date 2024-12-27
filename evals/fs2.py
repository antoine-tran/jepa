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
from fairseq2.assets import AssetCard, InProcAssetMetadataProvider, default_asset_store, setup_asset_store
from fairseq2.models.jepa import load_jepa_config

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
