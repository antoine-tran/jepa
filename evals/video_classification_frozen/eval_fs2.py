# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Test the parity of Fairseq2-ported V-JEPA models
# This script requires fairseq2 to be installed

from contextlib import AbstractContextManager, nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F

from fairseq2.logging import get_log_writer
from fairseq2.models.jepa import load_jepa_model
from fairseq2.models.jepa.classifier import JEPA_CLASSIFIER_FAMILY, load_jepa_classifier_model, JepaClassifierModel
from fairseq2.nn.utils.module import freeze_parameters, share_parameters, to_device
from fairseq2.recipes.utils.setup import setup_root_gang
from fairseq2.typing import CPU, Device, DataType

from evals.video_classification_frozen.utils import make_transforms
from src.utils.logging import AverageMeter
from src.datasets.data_manager import init_data
from src.utils.distributed import AllReduce

from evals.fs2 import Aggregator, create_model_card

log = get_log_writer(__name__)


def main(args_eval, resume_preempt=False):
    """Main entry to eval function.
    
    In contrast to the canonical JEPA evals, we
    only perform frozen-based evaluation without 
    any training. The `args_eval` also contains
    only 3 blocks: pretrain, data, eval
    
    Args:
        args-eval:
    """
        
    # -- PRETRAIN
    args_pretrain = args_eval.get("pretrain")
    model_name = args_pretrain.get("model_name")
    model_arch = args_pretrain.get("model_arch")
    tubelet_size = args_pretrain.get("tubelet_size", 2)
    
    # -- DATA
    args_data = args_eval.get('data')
    val_data_path = [args_data.get('dataset_val')]
    dataset_type = args_data.get('dataset_type', 'VideoDataset')
    resolution = args_data.get('resolution', 224)

    num_classes = args_data.get('num_classes')
    eval_num_segments = args_data.get('num_segments', 1)
    eval_frames_per_clip = args_data.get('frames_per_clip', 16)
    eval_frame_step = args_pretrain.get('frame_step', 4)
    eval_duration = args_pretrain.get('clip_duration', None)
    eval_num_views_per_segment = args_data.get('num_views_per_segment', 1)
    
    # -- EVAL
    args_eval = args_eval.get("eval")
    batch_size = args_eval.get("batch_size")
    classifier_checkpoint = Path(args_eval.get("checkpoint"))
    attend_across_segments = args_eval.get("attend_across_segments")
    
    # ----------------------------------------------------------------------- #

    gang = setup_root_gang(log)
    
    # Initialize model
    classifier_model_card = create_model_card(
        checkpoint=classifier_checkpoint,
        model_arch=model_arch,
        model_family=JEPA_CLASSIFIER_FAMILY,
        pretrain_model_card=model_name,
        num_classes=num_classes,
    )
    
    log.info(f"Model card: {classifier_model_card}")
    
    model = load_jepa_classifier_model(
        classifier_model_card,
        device=CPU,
        dtype=torch.float32,
        strict_state_dict=False,
    )
    
    if gang.rank == 0:
        pt_model = load_jepa_model(model_name, device=CPU, dtype=torch.float32)
        share_parameters(pt_model.encoder, model.encoder)
    
        del pt_model
        
        model = Aggregator(model, tubelet_size=tubelet_size, attend_across_segments=attend_across_segments)
        
        to_device(model, gang.device)

    gang.barrier()
    
    # TODO: Experiment with finetuning the classifier layer. For now we freeze everything
    # and perform pure evaluation
    model.eval()
    freeze_parameters(model)
    
    val_loader = make_dataloader(
        dataset_type=dataset_type,
        root_path=val_data_path,
        resolution=resolution,
        frames_per_clip=eval_frames_per_clip,
        frame_step=eval_frame_step,
        num_segments=eval_num_segments,
        eval_duration=eval_duration,
        num_views_per_segment=eval_num_views_per_segment,
        allow_segment_overlap=True,
        batch_size=batch_size,
        world_size=gang.size,
        rank=gang.rank,
    )
    
    val_acc = run_eval(
        model=model,
        data_loader=val_loader,
        device=gang.device,
        dtype=torch.float16,
        attend_across_segments=attend_across_segments,
    )

def make_dataloader(
    root_path,
    batch_size,
    world_size,
    rank,
    dataset_type='VideoDataset',
    resolution=224,
    frames_per_clip=16,
    frame_step=4,
    num_segments=8,
    eval_duration=None,
    num_views_per_segment=1,
    allow_segment_overlap=True,
    num_workers=4,
    subset_file=None
):
    # Make Video Transforms
    transform = make_transforms(
        training=False,
        num_views_per_clip=num_views_per_segment,
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(0.75, 4 / 3),
        random_resize_scale=(0.08, 1.0),
        reprob=0.25,
        auto_augment=True,
        motion_shift=False,
        crop_size=resolution,
    )

    data_loader, _ = init_data(
        data=dataset_type,
        root_path=root_path,
        transform=transform,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        clip_len=frames_per_clip,
        frame_sample_rate=frame_step,
        duration=eval_duration,
        num_clips=num_segments,
        allow_clip_overlap=allow_segment_overlap,
        num_workers=num_workers,
        copy_data=False,
        drop_last=False,
        subset_file=subset_file)
    return data_loader


def _maybe_autocast(device: Device, dtype: DataType) -> AbstractContextManager[None]:
    if dtype == torch.float32:
        return nullcontext()
    
    if dtype == torch.float16:
        if device.type == "cuda":
            return torch.cuda.amp.autocast(dtype=torch.float16, enabled=True)
        else:
            return torch.autocast(device_type=device.type, dtype=dtype, enabled=True)
    return torch.autocast(device_type=device.type, dtype=dtype)


def run_eval(
    model,
    data_loader,
    device,
    dtype,
    attend_across_segments=False,
):
    top1_meter = AverageMeter()
    for itr, data in enumerate(data_loader):

        with _maybe_autocast(device=device, dtype=dtype):

            # Load data and put on GPU
            clips = [
                [dij.to(device, non_blocking=True) for dij in di]  # iterate over spatial views of clip
                for di in data[0]  # iterate over temporal index of clip
            ]
            clip_indices = [d.to(device, non_blocking=True) for d in data[2]]
            labels = data[1].to(device)
            batch_size = len(labels)

            with torch.no_grad():
                outputs = model(clips, clip_indices=clip_indices)
        
        if attend_across_segments:
            outputs = sum([F.softmax(o, dim=1) for o in outputs]) / len(outputs)
            
        top1_acc = 100. * outputs.max(dim=1).indices.eq(labels).sum() / batch_size
        top1_acc = float(AllReduce.apply(top1_acc))
        top1_meter.update(top1_acc)

        if itr % 20 == 0:
            acc = top1_meter.avg
            mem = torch.cuda.max_memory_allocated() / 1024.**2
            log.info(f"[{itr:5d}] {acc:.3f}  [mem: {mem:.2e}]")

    return top1_meter.avg
