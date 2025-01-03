# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import pprint
import yaml

from evals.scaffold import main as eval_main

def process_main(fname: str, eval_module: str = "eval"):

    # Load config
    params = None
    with open(fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)

    # Launch the eval with loaded config
    eval_main(params['eval_name'], args_eval=params, eval_module=eval_module)


if __name__ == '__main__':
    from fire import Fire
    Fire(process_main)
