import numpy as np
import torch
import colorsys

def get_n_different_colors(n: int) -> np.ndarray:
    np.random.seed(0)
    return np.random.randint(0, 256, (n, 3), dtype=np.uint8)


def id2rgb(id, max_num_obj=256):
    if not 0 <= id <= max_num_obj:
        raise ValueError("ID should be in range(0, max_num_obj)")

    # Convert the ID into a hue value
    golden_ratio = 1.6180339887
    h = ((id * golden_ratio) % 1)           # Ensure value is between 0 and 1
    s = 0.5 + (id % 2) * 0.5       # Alternate between 0.5 and 1.0
    l = 0.5

    
    # Use colorsys to convert HSL to RGB
    rgb = np.zeros((3, ), dtype=np.uint8)
    if id==0:   #invalid region
        return rgb
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    rgb[0], rgb[1], rgb[2] = int(r*255), int(g*255), int(b*255)

    return rgb

def ndc2Pixel(v, S):
    return ((v + 1.0) * S - 1.0) * 0.5

def transformPoint4x4(point, matrix):
    """Transform a point by a 4x4 matrix.

    :param point: 3D point.
    :param matrix: 4x4 matrix.
    :return: Transformed point.
    """
    point = torch.cat([point, torch.ones_like(point[:, :1])], dim=1)
    transformed = torch.matmul(point, matrix)
    return transformed

def convert_matched_mask(labels, masks):
    assert labels.shape[0] == np.max(masks)
    matched_mask = np.zeros(masks.shape, dtype=np.uint16)
    for l in range(labels.shape[0]):
        matched_mask[masks == l+1] = labels[l].item() + 1

    return matched_mask.astype(np.uint8)

def mask_id_to_binary_mask(mask_id):
    num_masks = np.max(mask_id)
    h, w = mask_id.shape
    binary_mask = np.bool_(np.zeros((num_masks, h, w)))
    for m_idx in range(1, num_masks+1):
        binary_mask[m_idx-1] = (mask_id == m_idx)
    return binary_mask