import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
import torchvision
import json
from PIL import Image

from arguments import ModelParams, PipelineParams
from mask.utils import get_n_different_colors, ndc2Pixel, transformPoint4x4, convert_matched_mask, mask_id_to_binary_mask

from scene import Scene, GaussianModel

default_params = {
    "seg_method": "sam",
    "front_percentage": 0.2,
    "iou_threshold": 0.1,
    "num_patch": 32,
    "visualize": False
}

class GaussianProjector(torch.nn.Module):
    def __init__(self,
                 dataset : ModelParams,
                 pipeline : PipelineParams,
                 iteration : int,
                 params : dict = default_params,
                 device : torch.device = torch.device("cuda"),
                 ):
        super(GaussianProjector, self).__init__()
        self.device = device
        # Load pre-trained Gaussians and cameras
        self.gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, self.gaussians, load_iteration=iteration, shuffle=False)
        self.gaussians_xyz = self.gaussians.get_xyz.to(self.device)
        self.viewpoint_camera = scene.getTrainCameras() # Only use the training cameras for mask association
        # Key hyperparameters
        self.front_percentage = params["front_percentage"]
        self.iou_threshold = params["overlap_threshold"]
        self.num_patches = params["num_patch"]
        # Paths
        self.source_path = dataset.source_path
        self.seg_method = params["seg_method"]
        self.raw_mask_folder = os.path.join(self.source_path, "raw_{0}_mask".format(self.seg_method))
        assert os.path.exists(self.raw_mask_folder), "Mask folder does not exist."
        self.associated_mask_folder = os.path.join(self.source_path, "{0}_mask".format(self.seg_method))
        os.makedirs(self.associated_mask_folder, exist_ok=True)
        if params["visualize"]:
            self.visualize = True
            self.visualize_folder = os.path.join(self.source_path, "{0}_mask_vis".format(self.seg_method))
            os.makedirs(self.visualize_folder, exist_ok=True)
            self.random_colors = get_n_different_colors(1000)
        
        # For mask partition
        random_mask = self.load_mask(self.viewpoint_camera[0])
        self.image_width, self.image_height = random_mask.shape[1], random_mask.shape[2]
        self.patch_width = self.image_width // self.num_patches + 1 if self.image_width % self.num_patches != 0 else self.image_width // self.num_patches
        self.patch_height = self.image_height // self.num_patches + 1 if self.image_height % self.num_patches != 0 else self.image_height // self.num_patches
        self.patch_mask = torch.zeros((self.num_patches, self.num_patches, self.image_width, self.image_height), dtype=torch.bool, device=self.device)
        for i in range(self.num_patches):
            for j in range(self.num_patches):
                self.patch_mask[i, j, i*self.patch_width: (i+1)*self.patch_width, j*self.patch_height: (j+1)*self.patch_height] = True
        self.flatten_patch_mask = self.patch_mask.flatten(start_dim=2)

        # For mask association
        self.gaussian_idx_bank = []
        self.num_mask = 0
        self.assigned_gaussians = [] # We don't want the same Gaussian to be assigned to multiple masks

    @property
    def get_num_mask(self):
        if len(self.gaussian_idx_bank) == 0:
            self.num_mask = 0
            return 0
        self.num_mask = len(self.gaussian_idx_bank)
        return self.num_mask
    
    def maintain_gaussian_idx_bank(self, idx, front_gaussian_of_mask):
        assert not idx > self.num_mask, "idx is larger than the number of masks"
        if idx == self.num_mask:
            self.gaussian_idx_bank.append(front_gaussian_of_mask)
            self.assigned_gaussians = torch.unique(torch.cat([self.assigned_gaussians, front_gaussian_of_mask]))
        else:
            non_assigned_gaussians = torch.unique(front_gaussian_of_mask[~torch.isin(front_gaussian_of_mask, self.assigned_gaussians)])
            self.gaussian_idx_bank[idx] = torch.unique(torch.cat([self.gaussian_idx_bank[idx], non_assigned_gaussians]))
            self.assigned_gaussians = torch.unique(torch.cat([self.assigned_gaussians, non_assigned_gaussians]))
        
    def initialize(self, viewpoint):
        front_gaussian, mask = self.get_patch_front_gaussian_of_mask(viewpoint)
        self.gaussian_idx_bank.extend(front_gaussian)
        self.assigned_gaussians = torch.unique(torch.cat(front_gaussian))

        self.get_num_mask
        labels = torch.arange(self.num_mask, dtype=torch.long, device=self.device)

        return labels

    def associate(self, viewpoint):
        front_gaussian, mask = self.get_patch_front_gaussian_of_mask(viewpoint)
        num_mask_cur_view = len(front_gaussian)

        self.get_num_mask
        labels = torch.zeros(num_mask_cur_view, dtype=torch.long, device=self.device)
        for m_idx in range(num_mask_cur_view):
            front_gaussian_of_mask = front_gaussian[m_idx]
            num_union = [len(torch.unique(torch.cat([self.gaussian_idx_bank[i], front_gaussian_of_mask]))) for i in range(self.num_mask)]
            num_intersection = [len(self.gaussian_idx_bank[i]) + len(front_gaussian_of_mask) - num_union[i] for i in range(self.num_mask)]
            num_cur = len(front_gaussian_of_mask)
            # ### IOU
            # iou = [num_intersection[i] / (num_union[i] + 1e-8) for i in range(self.num_mask)]
            # iou = torch.tensor(iou, dtype=torch.float32, device=self.device)
            # selected_mask = torch.argmax(iou)
            # if iou[selected_mask] > self.iou_threshold:
            #     non_assigned_gaussians = torch.unique(front_gaussian_of_mask[~torch.isin(front_gaussian_of_mask, self.assigned_gaussians)])
            #     self.gaussian_idx_bank[selected_mask] = torch.unique(torch.cat([self.gaussian_idx_bank[selected_mask], non_assigned_gaussians]))
            ### IOCUR
            io_cur = [num_intersection[i] / (num_cur + num_intersection[i] + 1e-8) for i in range(self.num_mask)]
            io_cur = torch.tensor(io_cur, dtype=torch.float32, device=self.device)

            #!-------- Method 1: Pure IOU / IOCUR --------!#
            selected_mask = torch.argmax(io_cur)
            if io_cur[selected_mask] < self.iou_threshold:
                selected_mask = self.num_mask
            self.maintain_gaussian_idx_bank(selected_mask, front_gaussian_of_mask)
            # #!-------- Method 1: Pure IOU / IOCUR --------!#
            labels[m_idx] = selected_mask

            self.get_num_mask

        return labels

    def build_mask_association(self):
        for view in tqdm(self.viewpoint_camera):
            view = view.to(self.device)
            if self.num_mask == 0:
                labels = self.initialize(view)
            else:
                labels = self.associate(view)
            
            mask_path = os.path.join(self.raw_mask_folder, view.image_name + ".png")
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            object_mask = convert_matched_mask(labels, mask)
            object_mask_path = os.path.join(self.associated_mask_folder, view.image_name + ".png")
            cv2.imwrite(object_mask_path, object_mask)
            if self.visualize:
                visualize_mask = self.visualize_mask_association(object_mask)
                visualize_mask_path = os.path.join(self.visualize_folder, view.image_name + ".png")
                cv2.imwrite(visualize_mask_path, visualize_mask)
            # print("Number of masks: ", self.num_mask)

        info = {
            "num_mask": self.get_num_mask,
            "raw_mask_folder": self.raw_mask_folder,
            "associated_mask_folder": self.associated_mask_folder,
            "front_percentage": self.front_percentage,
            "iou_threshold": self.iou_threshold,
            "num_patch": self.num_patches,
        }
        json.dump(info, open(os.path.join(self.associated_mask_folder, "info.json"), "w"))

    def visualize_mask_association(self, object_mask):
        h, w = object_mask.shape
        visualize_mask = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(self.num_mask):
            visualize_mask[object_mask == i + 1] = self.random_colors[i]

        visualize_mask = cv2.cvtColor(visualize_mask, cv2.COLOR_RGB2BGR)
        return visualize_mask

    def project_gaussian(self, viewpoint):
        proj_matrix = viewpoint.full_proj_transform

        p_hom = transformPoint4x4(self.gaussians_xyz, proj_matrix)
        p_hom_z = p_hom[:, 2]

        p_w = 1 / (p_hom[:, 3:] + 1e-8)
        p_proj = p_hom[:, :3] * p_w

        p_proj[:, 0] = ndc2Pixel(p_proj[:, 0], self.image_width)
        p_proj[:, 1] = ndc2Pixel(p_proj[:, 1], self.image_height)
        p_proj = torch.round(p_proj[:, :2]).long()
        # Remove the points that are outside the image
        p_proj_inside_mask = (p_proj[:, 0] >= 0) & (p_proj[:, 0] < self.image_width) & (p_proj[:, 1] >= 0) & (p_proj[:, 1] < self.image_height) & (p_hom_z > 0)
        p_proj_inside = p_proj[p_proj_inside_mask]
        p_proj_inside_indices = p_proj_inside_mask.nonzero().squeeze()
        p_proj_inside_reverse_mapping = {p_proj_inside_indices[i].item(): i for i in range(len(p_proj_inside_indices))}
        p_proj_flatten = p_proj_inside[:, 0] * self.image_height + p_proj_inside[:, 1]

        projected_gaussian = {
            "p_proj_flatten": p_proj_flatten,
            "p_proj_inside_indices": p_proj_inside_indices,
            "p_proj_inside_reverse_mapping": p_proj_inside_reverse_mapping,
            "p_hom_z": p_hom_z
        }

        return projected_gaussian
    
    def load_mask(self, viewpoint):
        mask_path = os.path.join(self.raw_mask_folder, viewpoint.image_name + ".png")
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        binary_mask_mask = mask_id_to_binary_mask(mask)
        binary_mask = torch.tensor(binary_mask_mask, dtype=torch.bool, device=self.device).transpose(1, 2)
        return binary_mask
    
    def get_patch_front_gaussian_of_mask(self, viewpoint):
        projected_gaussian = self.project_gaussian(viewpoint)

        p_proj_flatten = projected_gaussian["p_proj_flatten"]
        p_proj_inside_indices = projected_gaussian["p_proj_inside_indices"]
        p_hom_z = projected_gaussian["p_hom_z"]

        mask = self.load_mask(viewpoint)
        # print("image shape: ", self.image_width, self.image_height)
        # if (mask.shape[1] != self.image_width) or (mask.shape[2] != self.image_height):
        #     mask = mask[:, :self.image_width, :self.image_height]
        assert mask.shape[1] == self.image_width and mask.shape[2] == self.image_height, "Mask and image have different sizes."
        mask_flatten = mask.flatten(start_dim=1)

        front_gaussian = []
        for obj_m in mask_flatten:
            front_gaussian_of_mask = []
            for i in range(self.num_patches):
                for j in range(self.num_patches):
                    patch_m = self.flatten_patch_mask[i, j]
                    m = obj_m & patch_m
                    if m.sum() == 0:
                        continue
                    gaussian_of_mask_inside = m[p_proj_flatten].nonzero().squeeze(-1)
                    if gaussian_of_mask_inside.shape[0] == 0:
                        continue
                    # print("gaussian_of_mask_inside: ", gaussian_of_mask_inside)
                    gaussian_of_mask = p_proj_inside_indices[gaussian_of_mask_inside]
                    p_hom_z_of_mask = p_hom_z[gaussian_of_mask]
                    num_front_gaussians = max(int(self.front_percentage * len(gaussian_of_mask)), 1)
                    front_gaussian_of_mask.append(gaussian_of_mask[torch.argsort(p_hom_z_of_mask)][:num_front_gaussians])
            if len(front_gaussian_of_mask) == 0:
                front_gaussian.append(torch.tensor([], dtype=torch.long, device=self.device))
            else:
                front_gaussian.append(torch.cat(front_gaussian_of_mask))

        return front_gaussian, mask