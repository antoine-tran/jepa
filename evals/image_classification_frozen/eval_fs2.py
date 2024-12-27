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

from fairseq2.logging import get_log_writer
from fairseq2.models.jepa import load_jepa_model
from fairseq2.models.jepa.classifier import JEPA_CLASSIFIER_FAMILY, load_jepa_classifier_model
from fairseq2.models.sequence import SequenceBatch
from fairseq2.nn.utils.module import freeze_parameters, share_parameters, to_device
from fairseq2.recipes.utils.setup import setup_root_gang
from fairseq2.typing import CPU, Device, DataType

import torchvision.transforms as transforms

from src.utils.logging import AverageMeter
from src.datasets.data_manager import init_data
from src.utils.distributed import AllReduce

from evals.fs2 import create_model_card

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
    
    # -- DATA
    args_data = args_eval.get('data')
    dataset_name = args_data.get('dataset_name')
    num_classes = args_data.get('num_classes')
    root_path = args_data.get('root_path', None)
    image_folder = args_data.get('image_folder', None)
    resolution = args_data.get('resolution', 224)
    
    # -- EVAL
    args_eval = args_eval.get("eval")
    batch_size = args_eval.get("batch_size")
    classifier_checkpoint = Path(args_eval.get("checkpoint"))
    
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
        
    pt_model = load_jepa_model(model_name, device=CPU, dtype=torch.float32)
    share_parameters(pt_model.encoder, model.encoder)
    
    del pt_model
    
    if gang.rank == 0:
        to_device(model, gang.device)
    
    gang.barrier()
    
    # TODO: Experiment with finetuning the classifier layer. For now we freeze everything
    # and perform pure evaluation
    model.eval()
    freeze_parameters(model)
    
    val_loader = make_dataloader(
        dataset_name=dataset_name,
        root_path=root_path,
        resolution=resolution,
        image_folder=image_folder,
        batch_size=batch_size,
        world_size=gang.size,
        rank=gang.rank,
        training=False,
    )
    
    val_acc = run_eval(
        model=model,
        data_loader=val_loader,
        device=gang.device,
        dtype=torch.float32,
    )

def make_dataloader(
    dataset_name,
    root_path,
    image_folder,
    batch_size,
    world_size,
    rank,
    resolution=224,
    training=False,
    subset_file=None
):
    normalization = ((0.485, 0.456, 0.406),
                     (0.229, 0.224, 0.225))
   
    transform = transforms.Compose([
        transforms.Resize(size=int(resolution * 256 / 224)),
        transforms.CenterCrop(size=resolution),
        transforms.ToTensor(),
        transforms.Normalize(normalization[0], normalization[1])])

    data_loader, _ = init_data(
        data=dataset_name,
        transform=transform,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        root_path=root_path,
        image_folder=image_folder,
        training=training,
        copy_data=False,
        drop_last=False,
        subset_file=subset_file)
    return data_loader


def _maybe_autocast(device: Device, dtype: DataType) -> AbstractContextManager[None]:
    if dtype == torch.float32:
        return nullcontext()
    
    if dtype == torch.float16:
        return torch.autocast(device_type=device.type, dtype=dtype, enabled=torch.bfloat16)
    
    return torch.autocast(device_type=device.type, dtype=dtype)


def run_eval(
    model,
    data_loader,
    device,
    dtype,
):
    top1_meter = AverageMeter()
    for itr, data in enumerate(data_loader):

        with _maybe_autocast(device=device, dtype=dtype):

            imgs, labels = data[0].to(device), data[1].to(device)
            batch = SequenceBatch(seqs=imgs, padding_mask=None)
            with torch.no_grad():
                outputs = model(batch)

        top1_acc = 100. * outputs.max(dim=1).indices.eq(labels).sum() / len(imgs)
        top1_acc = float(AllReduce.apply(top1_acc))
        top1_meter.update(top1_acc)

        if itr % 20 == 0:
            acc = top1_meter.avg
            mem = torch.cuda.max_memory_allocated() / 1024.**2
            log.info(f"[{itr:5d}] {acc:.3f}  [mem: {mem:.2e}]")

    return top1_meter.avg
