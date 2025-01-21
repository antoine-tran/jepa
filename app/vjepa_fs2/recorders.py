# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from collections.abc import MutableMapping
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import torch
from fairseq2.gang import Gang
from fairseq2.logging import get_log_writer
from fairseq2.metrics import (
    MetricBag,
    format_as_float,
    format_as_int,
    format_as_seconds,
)
from fairseq2.metrics.recorder import (
    MetricRecorder,
    _metric_formatters,
    register_metric_formatter,
)
from fairseq2.typing import override
from torch import Tensor
from torch.cuda import _get_device_index
from torcheval.metrics import Max, Mean, Sum, Throughput

logger = get_log_writer(__name__)

format_as_percent = partial(format_as_int, postfix="%")



## Weight and Biases recorder

try:
    import wandb  # type: ignore[import-not-found]
except ImportError:
    has_wandb = False
else:
    has_wandb = True


class LCMWandBRecorder(MetricRecorder):
    """Records metric values to Weights & Biases."""

    defined_runs: Set[str] = set()

    def __init__(
        self,
        project: Optional[str] = None,
        name: Optional[str] = None,
        output_dir: Optional[Path] = None,
        config: Dict[str, Any] = {},
        **kwargs,
    ) -> None:
        """
        :param project: A project to organise this run with other experiments, if none, the run will go under `uncategorized`.
        :param name: A unique name for your run, if none is given, a random name will be generated
        :param output_dir: The base directory under which to store the W&B files. You don't have to provide this.
        :param config: A dictionary of key-value pairs to be stored as the experiment's config. (akin to hparams in tb)
        :param kwargs: Additional arguments to pass to wandb.init()

        In order to use W&B, run `wandb login` from the command line and enter
        the API key when prompted.
        """
        if not has_wandb:
            log = get_log_writer(__name__)
            log.warning("wandb not found. Please install it with `pip install wandb`.")  # fmt: skip

            self._run = None
        else:
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
            self._run = wandb.init(  # type: ignore
                project=project,
                name=name,
                dir=output_dir,
                resume="allow",
                config=config,
                **kwargs,
            )

    def _define_run(self, run: str):
        if run in self.defined_runs:
            return
        # https://docs.wandb.ai/guides/track/log/customize-logging-axes/
        wandb.define_metric(f"{run}/step")
        wandb.define_metric(f"{run}/*", step_metric=f"{run}/step")

    @override
    def record_metrics(
        self,
        run: str,
        values: Mapping[str, Any],
        step_nr: Optional[int] = None,
        *,
        flush: bool = True,
    ) -> None:
        if self._run is None:
            return

        self._define_run(run)

        for name, value in values.items():
            formatter = _metric_formatters.get(name)
            if formatter is None:
                display_name = name
            else:
                display_name = formatter.display_name

            self._run.log({f"{run}/{display_name}": value, f"{run}/step": step_nr})

    @override
    def close(self) -> None:
        if self._run is not None:
            self._run.finish()


lcm_metric_formatters: Dict[str, Tuple[str, int, Callable[[Any], str]]] = {
    # fmt: off
    "loss": ("Loss", 100, format_as_float),
    "nll_loss": ("NLL Loss", 100, format_as_float),
    "mse_loss": ("MSE Loss", 100, format_as_float),
    "contrastive_loss": ("Contrastive Loss", 110, format_as_float),
    "reconstruction_loss": ("Reconstruction loss", 110, format_as_float),
    "unnormalized_reconstruction_loss": (
        "Unnormalized Reconstruction Loss",
        110,
        format_as_float,
    ),
    "kld": ("KLD loss", 110, format_as_float),
    "encoder_mse_loss": ("Encoder MSE loss", 110, format_as_float),
    "decoder_ce_loss": ("Decoder CE loss", 110, format_as_float),
    "elapsed_time": ("Elapsed Time", 500, format_as_seconds),
    "wall_time": ("Wall Time", 510, format_as_seconds),
    "lr": ("Learning Rate", 800, format_as_float),
    "loss_scale": ("Loss Scale", 810, format_as_float),
    "grad_norm": ("Grad norm", 810, format_as_float),
    "raw_grad_norm": ("Raw Grad norm", 815, format_as_float),
    "encoder_mse_scale": ("Encoder MSE loss scale", 850, format_as_float),
    "batch_size": ("Batch Size", 900, format_as_int),
    "elements_per_batch": ("Elements per Batch", 900, format_as_int),
    "elements_per_second": ("Elements per Second", 900, format_as_int),
    "num_examples": ("Number of Examples", 900, format_as_int),
    "num_source_elements": ("Number of Source Elements", 900, format_as_int),
    "num_target_elements": ("Number of Target Elements", 900, format_as_int),
    "total_num_target_elements": ("Accumulated Target Elements", 920, format_as_int),
    "gpu_memory_usage": ("GPU memory usage (GiB)", 910, format_as_float),
    "gpu_peak_memory_usage": ("GPU peak memory usage (GiB)", 910, format_as_float),
    "ram_percentage": ("RAM usage", 920, format_as_percent),
    "cpu_percentage": ("CPU usage", 920, format_as_percent),
    "mean_predicted_embeddings": ("mean_predicted_embeddings", 920, format_as_float),
    "std_predicted_embeddings": ("std_predicted_embeddings", 920, format_as_float),
    # fmt: on
}
for key in lcm_metric_formatters:
    register_metric_formatter(key, *lcm_metric_formatters[key], overwrite=True)
