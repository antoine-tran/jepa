
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import gc
import logging
import os
import sys
from abc import abstractmethod
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from functools import cached_property
from itertools import count
from pathlib import Path
from pprint import pformat
from typing import (
    Any,
    ContextManager,
    Dict,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Set,
    Tuple,
    override,
)

import torch
import yaml
from fairseq2.assets import AssetCard, AssetCardFieldNotFoundError
from fairseq2.checkpoint import FileCheckpointManager
from fairseq2.gang import FakeGang, Gang, ReduceOperation, all_sum
from fairseq2.logging import get_log_writer
from fairseq2.metrics import (
    LogMetricRecorder,
    MetricBag,
    MetricRecorder,
    TensorBoardRecorder,
    record_metrics,
)
from fairseq2.nn.ddp import to_ddp
from fairseq2.nn.fsdp import to_fsdp
from fairseq2.nn.utils.gradient import (
    check_gradient_norms,
    clip_gradient_norm,
    scale_gradients,
)
from fairseq2.nn.utils.module import (
    _get_named_modules,
    freeze_parameters,
    to_device,
)
from fairseq2.optim import AdamW, DynamicLossScaler
from fairseq2.utils.profiler import Profiler, Stopwatch
from fairseq2.utils.rng import RngBag
from fairseq2.utils.state import StatefulObjectBag
from fairseq2.optim.lr_scheduler import (
    AbstractLRScheduler,
    CosineAnnealingLR,
    MyleLR,
    NoopLR,
    PolynomialDecayLR,
    get_effective_lr,
    TriStageLR,
)
from torch.optim import Optimizer

from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel as FSDP,
)
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import record_function
from torcheval.metrics import Mean

from app.vjepa_fs2.configs import TrainingConfig
from app.vjepa_fs2.recorders import WandBRecorder
from app.vjepa_fs2.utils import flatten_dict


logger = get_log_writer(__name__)


def build_lr_scheduler(
    optimizer: Optimizer,
    lr: float,
    warmup_steps: int,
    start_lr: float = 1e-7,
    final_lr: float = 1e-5,
    max_steps: int = 10_000,
    stage_ratio: Tuple[float, ...] = (0.1, 0.4, 0.5),
    schedule: str = "myle",
) -> AbstractLRScheduler:
    assert schedule in [
        "noop",
        "myle",
        "cosine",
        "wsd",
        "polynomial",
    ], (
        f"Cannot recognize the learing rate schedule {schedule}, only noop, myle, cosine and wsd are supported"
    )

    assert lr > 0, "The learning reate should be strictly positive"

    lr_scheduler: AbstractLRScheduler

    if schedule == "noop":
        lr_scheduler = NoopLR(optimizer)

    elif schedule == "myle":
        lr_scheduler = MyleLR(
            optimizer,
            num_warmup_steps=warmup_steps,
            start_lr=[start_lr],
        )

    elif schedule == "cosine":
        lr_scheduler = CosineAnnealingLR(
            optimizer,
            cycle_len=max_steps - warmup_steps + 1,
            num_warmup_steps=warmup_steps,
            start_lr=[start_lr],
            final_lr=[final_lr],
            cycle_mul=1.0,
            lr_mul=1.0,
        )

    elif schedule == "wsd":
        assert lr > start_lr, (
            f"the starting learning rate {start_lr} should be lesser than the main lr {lr}"
        )
        start_lr_scale = start_lr / lr

        assert lr > final_lr, (
            f"the final learning rate {final_lr} should be lesser than the main lr {lr}"
        )
        final_lr_scale = final_lr / lr

        lr_scheduler = TriStageLR(
            optimizer,
            max_steps,
            stage_ratio=stage_ratio,  # type: ignore
            start_lr_scale=start_lr_scale,
            final_lr_scale=final_lr_scale,
        )

    elif schedule == "polynomial":
        lr_scheduler = PolynomialDecayLR(
            optimizer,
            max_steps,
            warmup_steps,
            power=200,
            start_lr=start_lr,
            final_lr=final_lr,
        )

    return lr_scheduler


class Trainer(StatefulObjectBag):
    config: TrainingConfig
    model: Module
    training_data_loader: Any
    validation_data_loader: Optional[Any]
    gang: Gang
    optimizer: Optimizer
    loss_scaler: DynamicLossScaler
    lr_scheduler: AbstractLRScheduler
    rng_bag: RngBag
    step_nr: int
    train_metric_bag: MetricBag
    valid_metric_bag: Mapping[str, MetricBag]
    metric_recorders: List[MetricRecorder]
    profiler: Profiler
    stopwatch: Stopwatch
    criterion: Any
    card_metdata: Dict
    _train_step_time: float
    _valid_step_time: float

    def __init__(
        self,
        config: TrainingConfig,
        model: Module,
        training_data_loader: Any,
        validation_data_loader: Optional[Any],
        gang: Gang,
        checkpoint_manager: FileCheckpointManager,
        rng_bag: RngBag,
        stopwatch: Stopwatch,
        card_metadata: Dict,
    ) -> None:
        super().__init__()

        self.config = config

        if self.config.debug:
            logger._logger.setLevel(logging.DEBUG)
            os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

        self.card_metadata = card_metadata

        self.dtype = eval(config.dtype)

        self.model = model

        self.training_data_loader = training_data_loader

        # Skip saving and loading the state of validation dataloader
        self.register_non_stateful("validation_data_loader", validation_data_loader)

        self.gang = gang

        self.rng_bag = rng_bag

        self.step_nr = 1

        self.current_run_steps = 0

        self.checkpoint_manager = checkpoint_manager

        tb_dir = config.tb_dir or config.output_dir.joinpath("tb")

        self.metric_recorders = [LogMetricRecorder(logger)]

        if gang.rank == 0:
            self.metric_recorders.append(TensorBoardRecorder(tb_dir))
            self.metric_recorders.append(
                WandBRecorder(
                    name=config.wandb_run_name,
                    project=config.wandb_project or "uncategorized",
                    output_dir=config.output_dir / "wandb",
                    config=self._tb_flat_config,
                )
            )

        self.profiler = Profiler(
            skip_first=config.profiler_skip_first,
            active=config.profiler_active,
            log_dir=tb_dir,
            gang=gang,
            enabled=config.profile,
        )

        self.stopwatch = stopwatch
        self._train_step_time = 0.0
        self._valid_step_time = 0.0

        self.criterion = None  # type: ignore

        self.loss_scaler = None  # type: ignore

    @property
    def is_fsdp(self) -> bool:
        return isinstance(self.model, FSDP)

    @property
    def is_ddp(self) -> bool:
        return isinstance(self.model, DDP)

    def setup(self) -> None:
        self.criterion = self.setup_criterion()

        self.setup_metric_bags()

        # Add the grad_norm metric to the training metric bag
        self.train_metric_bag.register_metric(
            "grad_norm", Mean(device=self.gang.device), persistent=False
        )
        self.train_metric_bag.register_metric(
            "raw_grad_norm", Mean(device=self.gang.device), persistent=False
        )

        self.setup_optimizer_and_lr_schedule()

    def setup_optimizer_and_lr_schedule(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            betas=tuple(self.config.adam_betas),  # type: ignore
            eps=self.config.adam_eps,
            use_fp32=self.config.use_optimizer_in_fp32,
            weight_decay=self.config.weight_decay,
        )
        logger.info(
            (
                f"Setting up AdamW optimizer with betas={self.config.adam_betas}, "
                f"base lr={self.config.lr} and weight decay={self.config.weight_decay} "
                f"and use_fp32={self.config.use_optimizer_in_fp32}"
            )
        )

        self.register_stateful("optimizer", optimizer)

        self.loss_scaler = DynamicLossScaler(
            optimizer,
            gang=self.gang,
            init_scale=self.config.loss_scaler_init_scale,
            min_scale=0.0001,
            scale_window=self.config.loss_scaler_scale_window,
            enabled=self.dtype == torch.float16,
        )

        if self.loss_scaler.is_enabled:
            logger.info(
                f"Initializing DynamicLossScaler with init_scale={self.config.loss_scaler_init_scale}"
            )

        lr_scheduler = build_lr_scheduler(
            optimizer=self.optimizer,
            schedule=self.config.lr_schedule,
            lr=self.config.lr,
            warmup_steps=self.config.num_lr_warmup_steps,
            start_lr=self.config.start_lr,
            final_lr=self.config.final_lr,
            max_steps=self.config.max_steps,
            stage_ratio=tuple(self.config.lr_stage_ratios),
        )

        # Saving the lr_scheduler as well to properly resume training
        self.register_stateful("lr_scheduler", lr_scheduler)

    @abstractmethod
    def setup_criterion(self):
        """Define a criterion (loss / objective function to optimize)"""

    def setup_metric_bags(self):
        """Setup metric bags for tracking"""

        self.train_metric_bag = MetricBag(self.gang)

        self.register_non_stateful(
            "valid_metric_bag",
            {
                # ds_name(dataset): MetricBag(self.gang)
                dataset: MetricBag(self.gang)
                for dataset in self.config.validation_data
            },
        )

    def checkpoint_and_raise(self, exc) -> None:
        # Checkpoint before exiting
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        logger.warning(f"R{self.gang.rank} checkpoint_and_raise - error={exc}")
        if self.current_run_steps > 100:
            # avoid checkpoining for early failures
            self._checkpoint(crash=exc)
        raise exc

    @cached_property
    def _tb_flat_config(self):
        """
        Prepare the flat config that will be used as HParams
        to record training metadata, namely config and environment hashes.
        """

        dict_config = flatten_dict(asdict(self.config))

        # Merge the data lists:
        def get_data_signature(dataset):
            return ":".join(
                map(str, (dataset["name"], dataset["weight"], dataset["filters"]))
            )

        dict_config["training_data"] = "+".join(
            get_data_signature(dataset) for dataset in dict_config["training_data"]
        )
        dict_config["validation_data"] = "+".join(
            get_data_signature(dataset) for dataset in dict_config["validation_data"]
        )

        # value should be one of int, float, str, bool, or torch.Tensor
        allowed_types = (int, float, str, bool, torch.Tensor)
        config_keys = list(dict_config)
        for k in config_keys:
            if not isinstance(dict_config[k], allowed_types):
                del dict_config[k]

        return dict_config

    def run(self) -> None:
        """Run the trainer for up to `max_steps`"""

        logger.info(f"Running training on {self.gang.size} device(s).")

        data_iter = self.training_data_loader.iterate_batches()

        logger.info(
            f"R{self.gang.rank} - waiting for all ranks to prepare a data iterator!"
        )
        self.gang.barrier()

        # These counters are rank-specific
        ooms, nans_or_infs = 0, 0

        # TODO: validate before training
        # logger.info(f"Starting with validation at step={self.step_nr}")
        # self._validate()

        with self.profiler:
            while self.step_nr <= self.config.max_steps:
                with record_function(f"step_{self.step_nr}"):
                    try:
                        # Main training step: forward -> backward -> optimizer.step -> log
                        stepped = self._train_step(data_iter)

                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            self._log_oom(e)
                            ooms += 1
                            if self.config.raise_oom or ooms > self.config.max_ooms:
                                # Previous behaviour, no retries but still checkpointing
                                self.checkpoint_and_raise(e)

                            logger.warning(
                                f"Attempting to recover from OOM on R{self.gang.rank} (OOMS={ooms})"
                            )
                            stepped = True
                            # reset optimizer
                            self.optimizer.zero_grad(set_to_none=True)

                            # rollback updates
                            self.train_metric_bag.rollback_updates()

                            # Empty CUDA cache before trying again
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                        else:
                            # Other RuntimeErrors
                            self.checkpoint_and_raise(e)

                    except FloatingPointError as e:
                        if "Losses are Nan/Inf" in str(e):
                            self._log_nan_loss(e)
                            nans_or_infs += 1
                            if (
                                self.config.raise_nan_or_inf
                                or nans_or_infs > self.config.max_nans_or_infs
                            ):
                                self.checkpoint_and_raise(e)

                            logger.warning(
                                f"Attempting to recover from NaN/Inf loss on R{self.gang.rank} (NaNs/Infs={nans_or_infs})"
                            )
                            stepped = True
                            # reset optimizer
                            self.optimizer.zero_grad(set_to_none=True)

                            # rollback updates
                            self.train_metric_bag.rollback_updates()

                        else:
                            # Other FloatingPointErrors
                            self.checkpoint_and_raise(e)

                    except Exception as e:
                        self.checkpoint_and_raise(e)

                if stepped:
                    if self._should_publish_train_metrics():
                        self._publish_train_metrics()

                    if self._should_checkpoint():
                        self._checkpoint()

                    if self._should_validate():
                        self._validate()

                    if self._should_collect_garbage():
                        self._collect_garbage()

                    self.profiler.step()

                    self.step_nr += 1
                    self.current_run_steps += 1

                else:
                    logger.info(f"R{self.gang.rank} - Resetting the datapipeline")
                    self.training_data_loader.pipeline.reset()

                    logger.info(f"R{self.gang.rank} - Done resetting the datapipeline")
                    data_iter = self.training_data_loader.iterate_batches()

        self._save_model_card_for_last_checkpoint(to_checkpoint_dir=False)
        logger.info(f"Finished training after {self.step_nr - 1} step(s).")

        self.gang.close()

    def restore(self) -> None:
        logger.info("Attempting to load last checkpoint.")

        step_nr, checkpoint = self.checkpoint_manager.load_last_checkpoint()

        logger.info(f"Checkpoint loaded, restoring training from step {step_nr}.")

        self.load_state_dict(checkpoint)

        self.gang.barrier()

        logger.info("Training restored, resuming.")

        self.step_nr = step_nr + 1

    def _maybe_with_autocast(self) -> ContextManager[None]:
        # autocast is only needed if training with mixed precision.
        # If training fails without it, check if some module with its weights
        # is not properly cast
        if self.config.use_autocast:
            return torch.autocast(device_type="cuda", dtype=self.dtype)
        else:
            return nullcontext()

    def _train_step(self, data_iter: Iterator) -> bool:
        step_nr = self.step_nr

        step_stopwatch = Stopwatch(start=True, device=self.gang.device)

        stepped = False

        # We have to retry the step in case of a gradient overflow.
        while not stepped:
            batches = []

            # Collect batches.
            with record_function(f"step_{step_nr}_data_load"):
                for _ in range(self.config.gradient_accumulation):
                    try:
                        batches.append(next(data_iter))
                    except StopIteration:
                        break

            if len(batches) != self.config.gradient_accumulation:
                logger.info(
                    f"R{self.gang.rank} -End of data reached at training step {step_nr}."
                )

                return False

            # create a copy of the current metrics
            # any update to the metrics from this point will either be committed with `commit_updates`
            # or ignored with `rollback_updates`
            self.train_metric_bag.begin_updates()

            num_targets = 0

            # Accumulate gradients.
            for batch_nr, batch in enumerate(batches):
                with self._maybe_no_sync(batch_nr, len(batches)):
                    with record_function(f"step_{step_nr}_{batch_nr}_forward"):
                        # autocast should wrap only the forward pass(es)
                        # of your network, including the loss computation(s).
                        # Backward passes under autocast are not recommended.
                        with self._maybe_with_autocast():
                            loss = self.criterion(batch)

                    if not (
                        torch.isfinite(loss.value).all() or self.loss_scaler.is_enabled
                    ):
                        raise FloatingPointError("Losses are Nan/Inf.")

                    # update metrics
                    self.train_metric_bag.update([loss])

                    with record_function(f"step_{step_nr}_{batch_nr}_backward"):
                        self.loss_scaler.backward(loss.value)

                    num_targets += loss.num_target_elements

            # Record and clip gradient norm
            grad_norm, raw_grad_norm = self.process_gradients(step_nr, num_targets)

            # Update parameters.
            with record_function(f"step_{step_nr}_optimizer"):
                # scale_result: LossScaleResult(old_scale: float, new_scale: float, overflow: bool, min_reached: bool)
                _, scale_result = self.loss_scaler.run_optimizer_step(step_nr)

            if scale_result.overflow:
                # Walk back the metrics update:
                self.train_metric_bag.rollback_updates()
                logger.debug(
                    f"R{self.gang.rank} rolled back update {self.train_metric_bag._original_metrics is None}"
                )

                if scale_result.min_reached:
                    logger.error(f"Loss has started exploding at step {step_nr}. Stopping training.")  # fmt: skip

                    raise FloatingPointError("The training loss has exploded.")

                logger.debug(f"Repeating training step {step_nr}.")

            else:
                self.lr_scheduler.step()

                stepped = True

            # Reset.
            self.optimizer.zero_grad(set_to_none=True)

        # Stepped = True:
        with record_function(f"step_{step_nr}_metrics"):
            # do something with losses and grad_norm

            self.train_metric_bag.commit_updates()

            # gradient norm is common to workers
            self.train_metric_bag.grad_norm.update(grad_norm)
            self.train_metric_bag.raw_grad_norm.update(raw_grad_norm)

            if self.gang.rank == 0:
                # update elapsed time once
                self._train_step_time += step_stopwatch.get_elapsed_time()

        del batches
        return stepped

    def _maybe_no_sync(self, batch_nr: int, num_batches: int) -> ContextManager[None]:
        if batch_nr < num_batches - 1 and self.gang.size > 1:
            return self.model.no_sync()
        return nullcontext()

    def normalize_gradients(self, num_targets: int) -> None:
        """
        :param num_target:
            The number of targets used in loss computation in this process.

        If reduction = sum:
            similar to fairseq2's `normalize_gradients`, will normalize the gradients of the model by ``world_size/num_targets``.
        If reduction = mean:
            will simply multiply by world size i.e undo DDP/FSDP's default normalization
        """
        reduction = self.criterion.reduction
        if reduction == "sum":
            total_num_targets = torch.tensor(
                num_targets, device=self.gang.device, dtype=torch.int64
            )

            self.gang.all_reduce(total_num_targets, ReduceOperation.SUM)

            # Both DDP and FSDP divide gradients by the world size which we also undo.
            if total_num_targets > 0:
                grad_scale = self.gang.size / total_num_targets
            else:
                # If total_num_targets == 0, gradients will be zeroes anyway
                grad_scale = self.gang.size

        else:
            grad_scale = self.gang.size

        scale_gradients(self.model, grad_scale)

    def process_gradients(
        self, step_nr: int, num_targets: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with record_function(f"step_{self.step_nr}_process_grads"):
            # Normalize gradients
            """
            Normalize and clip the gradients
            """
            # this raw grad norm is only used for debugging
            raw_grad_norm = clip_gradient_norm(
                self.model,
                max_norm=None,
            )

            if not self.config.turn_off_grad_normalization:
                self.normalize_gradients(num_targets=num_targets)

            # undo the GradScaler's scaling before clipping
            self.loss_scaler.unscale_gradients_()

            # Clip gradients
            # If DDP, we use torch.nn.utils.clip_grad_norm_, if FSDP,
            # we use torch.distributed.fsdp.FullyShardedDataParallel.clip_grad_norm_
            # this method handles the fact that gradients might be sharded across ranks.
            grad_norm = clip_gradient_norm(
                self.model,
                max_norm=self.config.max_grad_norm,
            )

            # Check for gradient consistency across workers:
            if not check_gradient_norms(grad_norm, self.gang, step_nr):
                raise FloatingPointError(
                    f"The gradients are inconsistent between processes at step {step_nr}. Training cannot continue."
                )

        return grad_norm, raw_grad_norm

    def _should_validate(self) -> bool:
        return self._should_do(self.config.validate_every_n_steps)

    def _should_collect_garbage(self) -> bool:
        return self._should_do(self.config.gc_every_n_steps)

    def _collect_garbage(self):
        logger.info("Collecting garbage...")
        gc.collect()

    @torch.inference_mode()
    def _validate(self) -> None:
        gc.collect()
        torch.cuda.empty_cache()

        if self.validation_data_loader is None:
            logger.info("Skip validation as the data loader is empty")
            return

        self.model.eval()

        logger.info(f"Starting validation after step {self.step_nr}.")

        self.validation_data_loader.pipeline.reset()

        data_iter = self.validation_data_loader.iterate_batches()
        data_dummy_iter = self.validation_data_loader.iterate_dummy_batches()

        logger.info(f"R{self.gang.rank} done creating the validation data iterator")

        for step_nr in count(start=1):
            step_stopwatch = Stopwatch(start=True, device=self.gang.device)

            try:
                batch = next(data_iter)
                true_batch = 1
            except StopIteration:
                batch = next(data_dummy_iter)
                true_batch = 0

            total_nb_batches = all_sum(self.gang, true_batch)

            if bool(total_nb_batches == 0):
                break
            # we apply model for all workers to avoid process groups sync issues
            loss = self.criterion(batch)

            if true_batch:
                self._valid_step_time += step_stopwatch.get_elapsed_time()
                self.valid_metric_bag[batch.name].update([loss])

        self._publish_validation_metrics()

        logger.info(
            f"R{self.gang.rank} Validation complete in {step_nr} steps, resuming training."
        )

        self.model.train()

    def _should_publish_train_metrics(self) -> bool:
        return self._should_do(self.config.publish_metrics_every_n_steps)

    def _set_elements_per_second(
        self, metric_values: Dict[str, Any], elapsed_time: float
    ) -> None:
        try:
            num_elements = metric_values[self.criterion.throughput_metric_name]
        except KeyError:
            return

        if not isinstance(num_elements, (int, float, torch.Tensor)):
            return

        if elapsed_time == 0.0:
            metric_values["elements_per_second"] = 0.0
        else:
            metric_values["elements_per_second"] = num_elements / elapsed_time

    def _publish_train_metrics(self) -> None:
        values = self.train_metric_bag.sync_and_compute_metrics()

        self.train_metric_bag.reset_non_persistent_metrics()

        # Only rank-0 to record and publish
        # since sync_and_compute_metrics's recipient rank is 0
        if self.gang.rank != 0:
            return

        assert values is not None

        values["lr"] = get_effective_lr(self.lr_scheduler)

        self._set_elements_per_second(values, self._train_step_time)

        if self.loss_scaler.is_enabled:
            values["grad_scale"] = self.loss_scaler.get_scale()

        values["wall_time"] = self.stopwatch.get_elapsed_time()
        values["elapsed_time"] = self._train_step_time

        record_metrics(self.metric_recorders, "Train", values, self.step_nr)

        self._train_step_time = 0.0

    def _publish_validation_metrics(self) -> None:
        values = {}
        for name, metric_bag in self.valid_metric_bag.items():
            values[name] = metric_bag.sync_and_compute_metrics()
            metric_bag.reset_non_persistent_metrics()

        # Only rank-0 to record and publish
        if self.gang.rank != 0:
            return

        for name, val in values.items():
            assert val is not None
            self._set_elements_per_second(val, self._valid_step_time)
            val["elapsed_time"] = self._valid_step_time
            val["wall_time"] = self.stopwatch.get_elapsed_time()
            valid_name = f"Valid | {name}"
            record_metrics(self.metric_recorders, valid_name, val, self.step_nr)

        # reset timers
        self._valid_step_time = 0.0

    def _should_checkpoint(self) -> bool:
        return self._should_do(self.config.checkpoint_every_n_steps)

    def _should_save_consolidated_model(self) -> bool:
        return self.is_fsdp and self._should_do(self.config.save_model_every_n_steps)

    def _checkpoint(self, crash=None) -> None:
        logger.info(f"Saving checkpoint at step {self.step_nr}")
        checkpoint = self.state_dict()

        metadata = {
            "config": self.config,
            "crash": crash,
        }

        self.checkpoint_manager.begin_checkpoint(self.step_nr)

        if self.is_fsdp:
            replicated_keys = None
        elif self.is_ddp:
            # If we do not shard, save the model and the optimizer only on rank 0.
            replicated_keys = {"model", "optimizer"}
        else:
            replicated_keys = {"*"}

        self.checkpoint_manager.save_state(checkpoint, replicated_keys=replicated_keys)

        self.checkpoint_manager.save_metadata(metadata)

        if self._should_save_consolidated_model():
            self._save_consolidated_model()

        # Create a model card only after creating model.pt
        # i.e., regular checkpointing with DDP or after consolidation with FSDP
        if not self.is_fsdp:
            self._save_model_card_for_last_checkpoint(to_checkpoint_dir=True)

        self.checkpoint_manager.commit_checkpoint()

        # Note that this logic looks at saved directories regardless of
        # the nature of the checkpointing, consolidated or not
        if self.config.keep_last_n_checkpoints != -1:
            self.checkpoint_manager.keep_last_n_checkpoints(
                self.config.keep_last_n_checkpoints,
                preserve_model=self.config.preserve_consolidated_models,
            )

        logger.info(f"Checkpoint saved by worker @rank={self.gang.rank}")

    def _save_consolidated_model(self) -> None:
        logger.info(f"Saving consolidated model at step {self.step_nr}.")
        self.checkpoint_manager.save_consolidated_fsdp_model(self.model)
        self._save_model_card_for_last_checkpoint(to_checkpoint_dir=True)
        logger.info("Consolidated model saved.")

    def _should_do(self, n_step: int) -> bool:
        return self.step_nr % n_step == 0

    def create_model_card_for_last_checkpoint(
        self, is_final: bool = False, **card_kwargs
    ) -> Optional[AssetCard]:
        """Create a model card based on the last saved checkpoint and the model config."""
        logger.warning(
            "Could not create a model card with a generic trainer.  Please use a model-specific one."
        )
        return None

    def _save_model_card_for_last_checkpoint(
        self, to_checkpoint_dir: bool = False
    ) -> None:
        """Save the model card for the last checkpoint to the checkpoint directory or the core output directory."""
        if self.gang.rank != 0:
            return

        if to_checkpoint_dir:
            current_step_nr = self.checkpoint_manager._checkpoint_step_nr
            output_dir = self.checkpoint_manager._checkpoint_dir.joinpath(
                f"step_{current_step_nr}.tmp"
            )
        else:
            output_dir = self.config.output_dir

        card = self.create_model_card_for_last_checkpoint(
            is_final=not to_checkpoint_dir
        )

        if card is not None:
            card_data = card._metadata  # TODO: use the exposed attribute when available
            with open(output_dir / "model_card.yaml", "w", encoding="utf-8") as outfile:
                yaml.dump(card_data, outfile, default_flow_style=False)
            logger.info(f"Model card saved in {output_dir}")

    def _log_oom(self, exc):
        logger.warning(
            f"OOM: Ran out of memory on R{self.gang.rank} with exception: {exc}"
        )

        if torch.cuda.is_available():
            for device_idx in range(torch.cuda.device_count()):
                logger.warning(torch.cuda.memory_summary(device=device_idx))

        sys.stderr.flush()

    def _log_nan_loss(self, exc):
        logger.warning(f"We hit a Nan/Inf Loss: raised with exception: {exc}")
