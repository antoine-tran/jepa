# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from omegaconf import MISSING


SUPPORTED_FSDP_MEMORY_POLICIES = Literal["standard", "low", "very_low"]
SUPPORTED_FSDP_WRAP_POLICIES = Literal["layer", "stack", "model"]


@dataclass
class DatasetConfig:
    """Holds the configuration for datasets, including transformation"""
    
    dataset_type: str = "vdeodataset"
    """Type of the datasets"""
    
    dataset_name_or_path: str
    """Name or path of the dataset"""
    
    weight: float = 1.0
    """
    Indicates relative weight of dataset that can be used for sampling from different datasets.
    """
    
@dataclass
class DataLoadingConfig:
    
    multiple_dataset_chaining: str = "sample"
    """
    This option allows to chain several datasets together.
    The chaining can be done in two ways:
    - `sample` : each dataset will be sampled with the provided weight
    - `concat` : datasets will be concatenated together (no weights taken into account)
    - `round_robin`: datasets will be sampled in a round robin fashion (no weights taken into account)
    """
    
    batch_size: Optional[int] = None
    """The output batch size."""

    order_by_length: bool = True
    """
    Whether to create the batches with homogeneous tokens length
    for more efficient padding.
    """
    


@dataclass
class TrainingConfig:
    """Holds the configuration of a training job."""

    training_data: Any = MISSING
    """The datasets to train with."""

    validation_data: Any = MISSING
    """The datasets to validate on."""

    model_arch: Optional[str] = None
    """Starting architecture for the model to train"""

    model_arch_overrides: Optional[Dict] = None
    """Dict of parameters to overwrite in `model_arch`"""

    model_config_or_name: Optional[Any] = None
    """The model configuration or name to train.
        This option cannot be paired with model_arch + model_arch_overrides
        If provided, this option supersedes model_arch + model_arch_overrides
    """
    output_dir: Path = MISSING
    """The output directory to store checkpoints and logs."""

    log_folder: Optional[Path] = None
    """The executor's log directory where stdout/stderr will be redirected.
        We will use this directory to optionally enable ATEN and NCCL
        logging (if debug is True) """

    tb_dir: Optional[Path] = None
    """The output directory to store tensorbaord logs"""

    # defaults to "uncategorized"
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_entity: Optional[str] = None

    dtype: str = "torch.float32"
    """The data type of the model."""

    lr_schedule: str = "myle"
    """The learning rate schedule out of
        `noop`: no learning rate schedule, just use the initial learning rate,
        `myle`: inv-sqrt as implemented in Fairseq,
        `cosine` cosine annealing schedule,
        `wsd` for  Warmup-Stable-Decay (WSD) or tri-stage """

    lr: float = 0.004
    """The initial (post-warm-up) learning rate for AdamW."""

    start_lr: float = 1e-7
    """The initial warmup learning rate."""

    final_lr: float = 1e-5
    """The final learning rate."""

    lr_stage_ratios: List[float] = field(default_factory=lambda: [0.1, 0.4, 0.5])
    """The ratios of the wsd (tri-stage) learning rate scheduler."""

    num_lr_warmup_steps: int = 800
    """The number of warm-up steps for the learning rate."""

    weight_decay: float = 0.1
    """The weight decay coefficient of AdamW (PyTorch default: 1e-2, Fs2 default: 0.0)."""

    adam_betas: List[float] = field(default_factory=lambda: [0.9, 0.98])
    """The beta coefficients of AdamW used for computing running averages of gradient and its square."""

    adam_eps: float = 1e-6
    """The term added to the denominator in AdamW to improve numerical stability.
        Default in FS2 and PyTorch is 1e-8. Previous hard coded value in our trainer is 1e-6"""

    use_optimizer_in_fp32: bool = True
    """if True, the optimizer (AdamW) will be initialized with `use_fp32 = True`
        i.e. we will store the optimizer state in single precision and convert
        gradients on-the-fly to single precision for numerical stability"""

    max_steps: int = 10_000
    """The maximum number of training steps."""

    max_grad_norm: float = 1000
    """Maximal gradient norm, for gradient clipping.
       gradients are multiplied by `torch.clamp(max_norm / (total_norm + 1e-6), max=1.0)`
       if max_norm is arbitrarily large, then we'll only report gradients norm
    """
    turn_off_grad_normalization: bool = False
    """If ``True``, Turn off gradient normalization"""

    gradient_accumulation: int = 1
    """The number of steps to accumulate gradients before an optimizer update."""

    validate_every_n_steps: int = 5000
    """The number of steps after which to validate the model."""

    checkpoint_every_n_steps: int = 5000
    """The number of steps after which to checkpoint."""

    keep_last_n_checkpoints: int = -1
    """The number of checkpoints to keep on disk."""

    save_model_every_n_steps: int = 5000
    """The number of steps after which to save a consolidated version of the model."""

    preserve_consolidated_models: bool = False
    """If `True`, only pt files excluding ones starting with `mdoel` will be deleted from the step checkpoint directory."""

    publish_metrics_every_n_steps: int = 1
    """The number of steps after which to publish training metrics."""

    gc_every_n_steps: int = 1000
    """The frequency of steps at which we collect garbage with `gc.collect()`."""

    seed: int = 2
    """The RNG seed to use while starting the job."""

    debug: bool = False
    """If ``True``, runs the trainer in debug mode"""

    profile: bool = False
    """If ``True``, runs the PyTorch profiler at the beginning of the training."""

    profiler_skip_first: int = 200

    profiler_active: int = 3
    """If profiling (``profile = True``), The profiler will skip the first ``skip_first`` steps, then do the active recording for the next ``active`` steps
    If planning to visualize the trace with tensorbaord, then ``active`` should be small (less than 10 steps), otherwise tb won't load!
    """
    loss_scaler_init_scale: float = 2.0**15
    """The initial scale for the gradient scaler, fairseq2's default is 2.0**15"""

    loss_scaler_scale_window: Optional[int] = None
    """The number of consecutive optimizer steps without inf/NaN gradients that must occur for the scale to be updated"""

    use_fsdp: bool = True
    """If ``True``, uses FSDP instead of DDP."""

    use_autocast: bool = False
    """If ``True``, wrap the forward pass in AMP autocast context.
        autocast is only needed if training with mixed precision.
        If training fails without it, check if some module with its weights is not properly cast
    """

    fsdp_wrap_granularity: SUPPORTED_FSDP_WRAP_POLICIES = "model"
    """The granularity at which to wrap the model."""

    fsdp_memory_policy: SUPPORTED_FSDP_MEMORY_POLICIES = "standard"
    """The FSDP memory policy."""

    fsdp_fp32_reduce: bool = False
    """ If ``True``, the gradients will be reduced in full precision even when dtype is `torch.float16`"""

    use_submitit: bool = True
    """If ``True``, setup the environment ti use submitit."""

    fake_gang_device: Optional[str] = None
    """If non-empty, the trainer will be set locally on a device, instead of distributed training."""

    experiment_name: Optional[str] = None
    """experiment name for job trackin, if None default to StopesModule naming"""

    raise_oom: bool = False
    """If ``True``, raise OOM errors when they occur, if ``False`` give it another try."""

    raise_nan_or_inf: bool = False
    """If ``True``, raise FloatingPointError with Nan/Inf losses, if ``False`` give it another try."""

    max_ooms: int = 10
    """If ```raise_oom`` is False, how many OOMs we can tolerate per rank before raising an error."""

    max_nans_or_infs: int = 10
    """If ```raise_nan_or_inf`` is False, how many Nan/Infs we can tolerate per rank before raising an error."""

    pretrained_model: Optional[str] = None
    """The model to finetune from."""

    pretrained_model_loader: Optional[str] = None
    """The function to load the pre-trained model"""

    ovewrite_config_with_pretrained: bool = True
    """If `True` and `pretrained_model` is not None, the the model_config
        will be overwritten by the loaded pretrained model config"""

    do_not_record_hparams: bool = False
    """If `True` skip recording hparams in TF"""

    freeze_modules: Optional[List[str]] = None
    """Name of modules in the model to be frozen when training/finetuning"""

    # Experimental
    freezing_strategy: Literal["none", "modules", "ffn", "ffn-adaln", "adaln"] = "none"
    """
    Freezing strategy to follow. Options are:
        1. none: Nothing will be frozen (default)
        2. modules: A list of modules to freeze will be read from `freeze_modules`
        3. ffn: All ffn sub-modules will be frozen
        4. ffn-adaln: all FFN and Adaln sub-modules will be frozen.
    """

