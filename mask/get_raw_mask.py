#
# Copyright (C) 2024, Gaga
# Gaga research group, https://github.com/weijielyu/Gaga
# All rights reserved.

import os
import cv2
import torch
import numpy as np
from typing import Dict
from argparse import ArgumentParser
from tqdm import tqdm
import json

def get_n_different_colors(n: int) -> np.ndarray:
    np.random.seed(0)
    return np.random.randint(1, 256, (n, 3), dtype=np.uint8)

def visualize_mask(mask: np.ndarray) -> np.ndarray:
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    num_masks = np.max(mask)
    random_colors = get_n_different_colors(num_masks)
    for i in range(num_masks):
        color_mask[mask == i+1] = random_colors[i]
    return color_mask

def get_seg_model(config: Dict, seg_method: str, device: str):
    if seg_method == "sam":
        from segment_anything import sam_model_registry
        from automatic_mask_generator import SamAutomaticMaskGenerator
        sam = sam_model_registry[config['sam_encoder_version']](checkpoint=config['sam_checkpoint_path']).to(device=device)
        auto_sam = SamAutomaticMaskGenerator(sam,
                                            points_per_side=config['sam_num_points_per_side'],
                                            points_per_batch=config['sam_num_points_per_batch'],
                                            pred_iou_thresh=config['sam_pred_iou_threshold'])
        return auto_sam
    # else: # seg_method == "entityseg"
    elif seg_method == "entityseg":
        from detectron2.config import get_cfg
        from detectron2.projects.deeplab import add_deeplab_config
        from detectron2.projects.CropFormer.mask2former import add_maskformer2_config
        from detectron2.projects.CropFormer.demo_cropformer.predictor import VisualizationDemo

        def setup_cfg(config):
            # load config from file and command-line arguments
            cfg = get_cfg()
            add_deeplab_config(cfg)
            add_maskformer2_config(cfg)
            config_file = config['entityseg_config_file']
            cfg.merge_from_file(config_file)
            entityseg_checkpoint_path = ['MODEL.WEIGHTS', config['entityseg_checkpoint_path']]
            cfg.merge_from_list(entityseg_checkpoint_path)
            cfg.freeze()
            return cfg

        entityseg_cfg = setup_cfg(config)
        entityseg_demo = VisualizationDemo(entityseg_cfg)

        return entityseg_demo

    # elif seg_method == "panopticseg":
    #     from detectron2.config import get_cfg
    #     from detectron2.projects.deeplab import add_deeplab_config
    #     from mask.Mask2Former.mask2former import add_maskformer2_config
    #     # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config
    #     # from detectron2.projects.CropFormer.demo_mask2former.predictor import VisualizationDemo
    #     from mask.Mask2Former.demo.predictor import VisualizationDemo
    #     def setup_cfg(config):
    #         # load config from file and command-line arguments
    #         cfg = get_cfg()
    #         add_deeplab_config(cfg)
    #         add_maskformer2_config(cfg)
    #         config_file = config['panopticseg_config_file']
    #         cfg.merge_from_file(config_file)
    #         panopticseg_checkpoint_path = ['MODEL.WEIGHTS', config['panopticseg_checkpoint_path']]
    #         cfg.merge_from_list(panopticseg_checkpoint_path)
    #         cfg.freeze()
    #         return cfg

    #     panopticseg_cfg = setup_cfg(config)
    #     panopticseg_demo = VisualizationDemo(panopticseg_cfg)

    #     return panopticseg_demo

    else:
        raise NotImplementedError

def get_sam_mask(auto_sam, image, confidence_threshold):
    mask_data = auto_sam.generate(image)

    pred_masks = mask_data['masks'].float()  # num masks * H * W
    pred_scores = mask_data['iou_preds']  # num masks * num masks

    # select by confidence threshold
    selected_indexes = (pred_scores >= confidence_threshold)
    selected_scores = pred_scores[selected_indexes]
    selected_masks  = pred_masks[selected_indexes]
    _, m_H, m_W = selected_masks.shape
    mask_id = np.zeros((m_H, m_W), dtype=np.uint8)

    # rank
    selected_scores, ranks = torch.sort(selected_scores)
    ranks = ranks + 1
    for index in ranks:
        mask_id[(selected_masks[index-1]==1).cpu().numpy()] = int(index)

    # Compress the masks
    mask_indices = np.unique(mask_id)
    cur_idx = 1
    output_mask = np.zeros((m_H, m_W), dtype=np.uint8)
    for idx in mask_indices:
        if idx == 0:
            continue
        mask = (mask_id == idx)
        if mask.sum() > 0 and (mask.sum() / selected_masks[idx-1].sum()) > 0.1:
            output_mask[mask] = cur_idx
            cur_idx += 1

    return output_mask

def get_entityseg_mask(seg_model, image, confidence_threshold):
    predictions = seg_model.run_on_image(image)
    pred_masks = predictions["instances"].pred_masks
    pred_scores = predictions["instances"].scores
    selected_indexes = (pred_scores >= confidence_threshold)
    selected_scores = pred_scores[selected_indexes]
    selected_masks  = pred_masks[selected_indexes]
    _, m_H, m_W = selected_masks.shape
    # print("m_H, m_W: ", m_H, m_W)
    mask_id = np.zeros((m_H, m_W), dtype=np.uint8)

    # rank
    selected_scores, ranks = torch.sort(selected_scores)
    ranks = ranks + 1
    for index in ranks:
        mask_id[(selected_masks[index-1]==1).cpu().numpy()] = int(index)

    return mask_id

# def get_panopticseg_mask(seg_model, image, confidence_threshold):
#     predictions, visualized_output = seg_model.run_on_image(image)
#     # print("predictions panoptic_seg: ", predictions["panoptic_seg"][0].shape)
#     # print("predictions instances: ", predictions["instances"])
#     # print("predictions sem_seg: ", predictions["sem_seg"])
#     # pred_masks = predictions["instances"].pred_masks
#     # pred_scores = predictions["instances"].scores
#     # selected_indexes = (pred_scores >= confidence_threshold)
#     # selected_scores = pred_scores[selected_indexes]
#     # selected_masks  = pred_masks[selected_indexes]
#     # _, m_H, m_W = selected_masks.shape
#     # print("m_H, m_W: ", m_H, m_W)
#     # mask_id = np.zeros((m_H, m_W), dtype=np.uint8)
#     mask_id = predictions["panoptic_seg"][0].cpu().numpy().astype(np.uint8)
#     return mask_id

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_folder", "-d", default="/ssd1/wjlyu/Gaga/dataset", type=str)
    parser.add_argument("--scene", "-s", default="mipnerf360/room", type=str)
    parser.add_argument("--image", "-i", default="images", type=str)
    parser.add_argument("--seg_method", "-m", default="sam", type=str)
    parser.add_argument("--visualize", "-v", action="store_true")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    # make output folders
    output_folder = os.path.join(args.dataset_folder, args.scene, "raw_{}_mask".format(args.seg_method))
    os.makedirs(output_folder, exist_ok=True)
    if args.visualize:
        vis_output_folder = os.path.join(args.dataset_folder, args.scene, "raw_{}_mask_vis".format(args.seg_method))
        os.makedirs(vis_output_folder, exist_ok=True)

    # locate image folder
    image_folder = os.path.join(args.dataset_folder, args.scene, args.image)
    assert os.path.exists(image_folder)

    # load config
    config = json.load(open(os.path.join(os.path.dirname(__file__), 'config.json'), 'r'))[args.seg_method]

    # load model
    print("Loading {} model...".format(args.seg_method))
    seg_model = get_seg_model(config, args.seg_method, device=device)

    # generate masks
    print("Generating masks...")
    for image_name in tqdm(sorted(os.listdir(image_folder))):
        image_path = os.path.join(image_folder, image_name)
        if args.seg_method == "sam":
            image = cv2.imread(image_path)
            mask = get_sam_mask(seg_model, image, config["confidence_threshold"])
        elif args.seg_method == "entityseg":
            from detectron2.data.detection_utils import read_image
            image = read_image(image_path, format="BGR")
            mask = get_entityseg_mask(seg_model, image, config["confidence_threshold"])
        # elif args.seg_method == "panopticseg":
        #     from detectron2.data.detection_utils import read_image
        #     image = read_image(image_path, format="BGR")
        #     mask = get_panopticseg_mask(seg_model, image, config["confidence_threshold"])
        else:
            raise NotImplementedError

        # mask_path = os.path.join(output_folder, image_name.replace(".jpg", ".png"))
        mask_path = os.path.join(output_folder, image_name.split(".")[0] + ".png")
        cv2.imwrite(mask_path, mask)

        if args.visualize:
            vis_mask = visualize_mask(mask)
            vis_mask_path = os.path.join(vis_output_folder, image_name)
            cv2.imwrite(vis_mask_path, vis_mask)