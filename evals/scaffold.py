# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import importlib
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def main(
    eval_name,
    args_eval,
    resume_preempt=False,
    fs2=False
):
    # eval_module = "eval_fs2" if fs2 else "eval"
    eval_module = "test_parity_patch_embed"
    logger.info(f'Running evaluation: {eval_name}')
    return importlib.import_module(f'evals.{eval_name}.{eval_module}').main(
        args_eval=args_eval,
        resume_preempt=resume_preempt)
