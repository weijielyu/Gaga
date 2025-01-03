#
# Copyright (C) 2024, Gaga
# Gaga research group, https://github.com/weijielyu/Gaga
# All rights reserved.
#

from argparse import ArgumentParser

from replica import process_replica
from scannet import process_scannet

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dataset_folder", default="/ssd1/wjlyu/Gaga/dataset", type=str)
    parser.add_argument("--dataset" , default="replica", type=str)  # replica or scannet
    parser.add_argument("--input_folder", default="/ssd1/wjlyu/Gaga/dataset/Replica_Dataset", type=str)

    args = parser.parse_args()

    if args.dataset == "replica":
        process_replica(args)
    elif args.dataset == "scannet":
        process_scannet(args)
    else:
        raise NotImplementedError
