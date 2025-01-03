#
# Copyright (C) 2024, Gaga
# Gaga research group, https://github.com/weijielyu/Gaga
# All rights reserved.
#

import os
import glob
from scannet.split import split_evenly
from scannet.preprocess import unzip_raw_2d_files, preprocess_imgs

def process_scannet(args):
    print("Loading ScanNet dataset...")
    input_folder = args.input_folder
    assert os.path.exists(input_folder), "ScanNet dataset not found."
    unsplit_output_folder = os.path.join(args.dataset_folder, "scannet_unsplit")
    os.makedirs(unsplit_output_folder, exist_ok=False)

    scenes = ["scene0010_00", "scene0012_00", "scene0033_00", "scene0038_00", "scene0088_00", "scene0113_00", "scene0192_00"]

    unzip_raw_2d_files(input_folder, unsplit_output_folder, scenes)

    scene_names = glob.glob(os.path.join(unsplit_output_folder, "*"))
    for scene_f in scene_names:
        preprocess_imgs(scene_f)

    output_folder = os.path.join(args.dataset_folder, "scannet")
    for scene in scenes:
        scene_f = os.path.join(unsplit_output_folder, scene)
        print("Splitting scene: ", scene)
        split_evenly(scene_f, 300, output_folder)

        # Colmap
        print("Processing Colmap...")
        os.system("python convert.py -s {} --no_gpu".format(os.path.join(output_folder, scene)))

    os.system("rm -r {}".format(unsplit_output_folder))
    print("ScanNet dataset processed.")