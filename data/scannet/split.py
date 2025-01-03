import numpy as np
import os
import shutil
import cv2

def crop(data, H, W, crop_h, crop_w):
    crop_size = [crop_w, crop_h]
    crop_mask = np.zeros(shape=(H, W))
    t_w, t_h = crop_size
    margin_h = (H - t_h) // 2
    margin_w = (W - t_w) // 2
    crop_mask[margin_h: (H - margin_h), margin_w: (W - margin_w)] = 1
    crop_mask = crop_mask.astype(np.int8)
    data = data[crop_mask == 1]
    return data.reshape([crop_h, crop_w])

def ins_npz_num(f):
    npz = np.load(f)
    ins_map = npz.f.ins_2d_label_id
    ins_map = cv2.resize(ins_map, (640, 480), interpolation=cv2.INTER_NEAREST)
    shape = ins_map.shape
    ins_map = crop(ins_map, shape[0], shape[1], 450, 600)
    labels = np.unique(ins_map)[1:]
    return len(labels)

class Split:
    def __init__(self, scene_path, save_scene, train_index, test_index):
        super(Split, self).__init__()
        self.scene_path = scene_path
        self.save_path = save_scene
        self.initial_files()
        self.train_index, self.test_index = train_index, test_index

    def initial_files(self):
        self.colors_file = os.path.join(self.scene_path, 'color')
        self.ins_full_file = os.path.join(self.scene_path, 'instance-filt')

        return

    def copy(self):
        # copy train data
        self.split_train_path = os.path.join(self.save_path, 'train')
        os.makedirs(self.split_train_path, exist_ok=True)

        img_save_file = os.path.join(self.split_train_path, 'train_images')
        ins_full_save_file = os.path.join(self.split_train_path, 'train_ins_full')

        os.makedirs(img_save_file, exist_ok=True)
        os.makedirs(ins_full_save_file, exist_ok=True)

        for i, index in enumerate(self.train_index):  # i means number, index means one of train_index

            img_source_file = os.path.join(self.colors_file, f'{index}.jpg')
            img_target_file = os.path.join(img_save_file, f'{index}.jpg')
            shutil.copy(img_source_file, img_target_file)

            ins_full_source_file = os.path.join(self.ins_full_file, f'{index}.png')
            ins_full_target_file = os.path.join(ins_full_save_file, f'{index}.png')
            shutil.copy(ins_full_source_file, ins_full_target_file)

        # copy test data
        self.split_test_path = os.path.join(self.save_path, 'test')
        os.makedirs(self.split_test_path, exist_ok=True)

        img_save_file = os.path.join(self.split_test_path, 'test_images')
        ins_full_save_file = os.path.join(self.split_test_path, 'test_ins_full')

        os.makedirs(img_save_file, exist_ok=True)
        os.makedirs(ins_full_save_file, exist_ok=True)

        for i, index in enumerate(self.test_index):  # i means number, index means one of train_index

            img_source_file = os.path.join(self.colors_file, f'{index}.jpg')
            img_target_file = os.path.join(img_save_file, f'{index}.jpg')
            shutil.copy(img_source_file, img_target_file)

            ins_full_source_file = os.path.join(self.ins_full_file, f'{index}.png')
            ins_full_target_file = os.path.join(ins_full_save_file, f'{index}.png')
            shutil.copy(ins_full_source_file, ins_full_target_file)

        # copy for colmap
        self.split_colmap_path = os.path.join(self.save_path, 'input')
        os.makedirs(self.split_colmap_path, exist_ok=True)

        for i, index in enumerate(self.train_index):
            img_source_file = os.path.join(self.colors_file, f'{index}.jpg')
            img_target_file = os.path.join(self.split_colmap_path, f'{index}.jpg')
            shutil.copy(img_source_file, img_target_file)
        
        for i, index in enumerate(self.test_index):
            img_source_file = os.path.join(self.colors_file, f'{index}.jpg')
            img_target_file = os.path.join(self.split_colmap_path, f'{index}.jpg')
            shutil.copy(img_source_file, img_target_file)

def split_evenly(scene, number, save_dir):
    scene_name = str.split(scene, "/")[-1]
    # statistic instances
    ins_basedir = scene + '/instance-filt-cls19'
    len_ins_files = len(os.listdir(ins_basedir))
    statistic_results = [ins_npz_num(os.path.join(ins_basedir, f'{ids}.npz')) for ids in range(0, len_ins_files)]
    statistic_results = np.array(statistic_results)
    val_ids = np.where(statistic_results != 0)[0]

    amounts = len(val_ids)

    step = amounts // number
    train_idx = list(range(0, amounts, step))
    train_ids = val_ids[train_idx]
    test_idx = np.array([x + step // 2 for x in train_idx if (x + step) < (amounts - 1)])

    margin = len(test_idx) - number + 100
    start = margin // 2
    end = len(test_idx) - start

    test_selected_idx = np.array(list(range(start, end, 2))).astype(int)
    test_idx = test_idx[test_selected_idx]
    test_ids = val_ids[test_idx]

    save_scene = os.path.join(save_dir, scene_name)
    if not os.path.exists(save_scene):
        os.makedirs(save_scene)
    save_train_ids = os.path.join(save_scene, 'train_split.txt')
    np.savetxt(save_train_ids, train_ids, fmt="%i", delimiter='\n')
    save_test_ids = os.path.join(save_scene, 'test_split.txt')
    np.savetxt(save_test_ids, test_ids, fmt="%i", delimiter='\n')

    s = Split(scene, save_scene, train_ids, test_ids)
    s.copy()