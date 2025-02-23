#
# Copyright (C) 2024, Gaga
# Gaga research group, https://github.com/weijielyu/Gaga
# All rights reserved.
#

import os

def process_replica(args):
    print("Loading Replica dataset...")

    input_folder = args.input_folder
    assert os.path.exists(input_folder), "Replica dataset not found."
    output_folder = os.path.join(args.dataset_folder, "replica")
    os.makedirs(output_folder, exist_ok=False)

    total_num = 900
    sample_step = 5
    train_ids = list(range(0, total_num, sample_step))
    test_ids = [x + sample_step // 2 for x in train_ids]

    scenes = ["office_0", "office_1", "office_2", "office_3", "office_4", "room_0", "room_1", "room_2"]
    for scene in scenes:
        print("Processing scene: ", scene)
        # Copy images
        print("Copying images...")
        os.makedirs(os.path.join(output_folder, scene, "input"), exist_ok=True)
        for idx in train_ids:
            os.system("cp {}/{}/Sequence_1/rgb/rgb_{}.png {}/{}/input/train_rgb_{:04d}.png".format(input_folder, scene, idx, output_folder, scene, idx))
        for idx in test_ids:
            os.system("cp {}/{}/Sequence_1/rgb/rgb_{}.png {}/{}/input/test_rgb_{:04d}.png".format(input_folder, scene, idx, output_folder, scene, idx))

        # Copy semantic instance masks
        os.makedirs(os.path.join(output_folder, scene, "semantic_instance"), exist_ok=True)
        os.makedirs(os.path.join(output_folder, scene, "vis_sem_instance"), exist_ok=True)
        for idx in train_ids:
            os.system("cp {}/Replica_Instance_Segmentation/{}/Sequence_1/semantic_instance/semantic_instance_{}.png {}/{}/semantic_instance/train_semantic_instance_{:04d}.png".format(input_folder, scene, idx, output_folder, scene, idx))
            os.system("cp {}/Replica_Instance_Segmentation/{}/Sequence_1/semantic_instance/vis_sem_instance_{}.png {}/{}/vis_sem_instance/train_vis_sem_instance_{:04d}.png".format(input_folder, scene, idx, output_folder, scene, idx))
        for idx in test_ids:
            os.system("cp {}/Replica_Instance_Segmentation/{}/Sequence_1/semantic_instance/semantic_instance_{}.png {}/{}/semantic_instance/test_semantic_instance_{:04d}.png".format(input_folder, scene, idx, output_folder, scene, idx))
            os.system("cp {}/Replica_Instance_Segmentation/{}/Sequence_1/semantic_instance/vis_sem_instance_{}.png {}/{}/vis_sem_instance/test_vis_sem_instance_{:04d}.png".format(input_folder, scene, idx, output_folder, scene, idx))

        # Colmap
        print("Processing Colmap...")
        os.system("python convert.py -s {} --no_gpu".format(os.path.join(output_folder, scene)))

    # Remove Replica_Dataset
    os.system("rm -r {}/Replica_Dataset".format(args.dataset_folder))
