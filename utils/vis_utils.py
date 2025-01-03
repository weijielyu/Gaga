import numpy as np

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