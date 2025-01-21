# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import MutableMapping
from typing import Dict, List


def flatten_dict(d: MutableMapping, parent_key: str = "", sep: str = ".") -> Dict:
    """
    A helper function to flatten nested dictionaries
    Example. With a training config like
        config = {
        'data': {
            'training': {'batch_size': 10},
            'validation': {'batch_size': 2}
            },
        'model': {'model_dim': 1024},
        'use_fsdp': True
        }
        The flat config will be:
            {
            'data.training.batch_size': 10,
            'data.validation.batch_size': 2,
            'model.model_dim': 1024,
            'use_fsdp': True
            }
        This helper is used to convert our nested training config into a flat
        dictionary for Tensoarboard's HParams conusmption

    """
    items: List = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
