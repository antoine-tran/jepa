# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


_GLOBAL_SEED = 0


def main(args, resume_preempt=False):
    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    meta_cfg = args.get("meta")

    # RNG seed to use while training
    seed: int = meta_cfg.get("seed", _GLOBAL_SEED)

    # If True, use FSDP instead of DD
    use_fsdp: bool = meta_cfg.get("use_fsdp", True)

    # If True, wrap the forward pass in AMP autocast context
    use_autocast: bool = meta_cfg.get("use_autocast", True)

    # If ``True``, runs the trainer in debug mode
    debug: bool = meta_cfg.get("debug", False)
    
    # The data type of the model
    dtype: str = meta_cfg.get("dtype", "torch.float32")
    
    # The number of steps after which to save a consolidated version of the model.
    save_model_every_n_steps = meta_cfg.get("save_model_every_n_steps", 5000)
    
    