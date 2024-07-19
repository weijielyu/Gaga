#
# Copyright (C) 2024, Gaga
# Gaga research group, https://github.com/weijielyu/Gaga
# All rights reserved.
#

import os
import numpy as np
from argparse import ArgumentParser
import cv2
import torch
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

def calculate_iou(mask1: torch.Tensor, mask2: torch.Tensor) -> float:
    """Helper function to calculate IoU between two masks."""
    intersection = torch.logical_and(mask1, mask2)
    union = torch.logical_or(mask1, mask2)
    assert torch.sum(union) > 0, "The union of the two masks must be non-zero"
    iou_score = torch.sum(intersection) / torch.sum(union)
    return iou_score

def get_iou_for_label_pair(pred_masks: torch.Tensor, gt_masks: torch.Tensor, gt_label_idx: torch.int, pred_label_idx: torch.int) -> torch.Tensor:
    """Calculate the IoU score for a pair of predicted and ground truth labels."""
    assert pred_masks.shape == gt_masks.shape, "Predicted and ground truth masks must have the same shape"
    all_image_iou = []
    for i in range(len(gt_masks)):
        gt_masks_binary = gt_masks[i] == gt_label_idx
        pred_masks_binary = pred_masks[i] == pred_label_idx
        if torch.sum(gt_masks_binary) == 0:
            continue
        iou = calculate_iou(pred_masks_binary, gt_masks_binary)
        all_image_iou.append(iou)

    return torch.tensor(all_image_iou).mean()

def get_linear_sum_assignment(iou_matrix: np.ndarray) -> np.ndarray:
    """Solve the linear sum assignment problem using the Hungarian algorithm."""
    row_ind, col_ind = linear_sum_assignment(iou_matrix, maximize=True)
    return row_ind, col_ind

if __name__ == "__main__":
    # Load the ground truth and predicted masks
    args = ArgumentParser()
    args.add_argument("--gt_masks", type=str, required=True, help="Path to the ground truth masks")
    args.add_argument("--pred_masks", type=str, required=True, help="Path to the predicted masks")
    args = args.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pred_mask_names = os.listdir(args.pred_masks)
    pred_masks = [cv2.imread(os.path.join(args.pred_masks, mask_name), cv2.IMREAD_UNCHANGED) for mask_name in pred_mask_names]
    gt_masks = []
    for mask_name in pred_mask_names:
        if "replica" in args.gt_masks:
            assert mask_name.startswith("test_rgb"), "The image name must start with 'test_rgb'"
            gt_mask_name = mask_name.replace("test_rgb", "test_semantic_instance")
            gt_mask = cv2.imread(os.path.join(args.gt_masks, gt_mask_name), cv2.IMREAD_UNCHANGED)
        else:
            gt_mask = cv2.imread(os.path.join(args.gt_masks, mask_name), cv2.IMREAD_UNCHANGED)
        gt_masks.append(gt_mask)

    pred_masks = np.array(pred_masks, dtype=np.int64)
    gt_masks = np.array(gt_masks, dtype=np.int64)
    pred_masks = torch.tensor(pred_masks).to(device)
    gt_masks = torch.tensor(gt_masks).to(device)

    num_gt_mask, h, w = gt_masks.shape
    num_pred_mask, h_pred, w_pred = pred_masks.shape

    assert num_gt_mask == num_pred_mask, "The number of ground truth masks must be equal to the number of predicted masks"

    if h != h_pred or w != w_pred:
        pred_masks = torch.nn.functional.interpolate(pred_masks.unsqueeze(0).float(), size=(h, w), mode="nearest").long().squeeze(0)

    gt_label_idx = torch.unique(gt_masks)
    num_gt_mask = len(gt_label_idx)

    pred_label_idx = torch.unique(pred_masks)
    num_pred_mask = len(pred_label_idx)

    print(f"Number of ground truth masks: {num_gt_mask}")
    print(f"Number of predicted masks: {num_pred_mask}")

    # Build IoU matrix
    iou_matrix = torch.zeros((num_gt_mask, max(num_gt_mask, num_pred_mask))).to(device)
    for i in tqdm(range(num_gt_mask)):
        for j in range(num_pred_mask):
            iou_matrix[i, j] = get_iou_for_label_pair(pred_masks, gt_masks, gt_label_idx[i], pred_label_idx[j])

    # Solve the linear sum assignment problem
    row_ind, col_ind = get_linear_sum_assignment(iou_matrix.cpu().numpy())

    # Get mean IoU, precision, and recall
    paired_iou = iou_matrix[row_ind, col_ind]
    mean_iou = paired_iou.mean()
    print(f"Mean IoU score: {mean_iou:.4f}")

    num_hit_05 = torch.sum(paired_iou > 0.5)
    precision_05 = num_hit_05 / num_pred_mask
    recall_05 = num_hit_05 / num_gt_mask

    print(f"Precision (IoU > 0.5): {precision_05:.4f}")
    print(f"Recall (IoU > 0.5): {recall_05:.4f}")
