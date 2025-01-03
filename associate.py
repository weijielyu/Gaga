#
# Copyright (C) 2024, Gaga
# Gaga research group, https://github.com/weijielyu/Gaga
# All rights reserved.
#

import os
import torch
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args

from mask.projector import GaussianProjector

if __name__ == "__main__":
    parser = ArgumentParser()
    # model = ModelParams(parser, sentinel=True)
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--seg_method", default="sam", type=str)
    parser.add_argument("--front_percentage", "-fp", type=float, default=0.2)
    parser.add_argument("--overlap_threshold", "-ot", type=float, default=0.1)
    parser.add_argument("--num_patch", "-np", type=int, default=32)
    parser.add_argument("--visualize", "-v", action="store_true")

    args = get_combined_args(parser)

    hyper_params = {
        "front_percentage": args.front_percentage,
        "overlap_threshold": args.overlap_threshold,
        "num_patch": args.num_patch,
        "seg_method": args.seg_method,
        "visualize": args.visualize
    }

    with torch.no_grad():
        projector = GaussianProjector(model.extract(args), pipeline.extract(args), args.iteration, hyper_params)
        projector.build_mask_association()