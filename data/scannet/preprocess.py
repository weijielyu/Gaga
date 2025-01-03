#
# Copyright (C) 2024, Gaga
# Gaga research group, https://github.com/weijielyu/Gaga
# All rights reserved.
#

import os
import csv
import copy
import glob
import shutil
import imageio
import numpy as np
from scannet.SensorData import SensorData

class Data_configs:
    sem_names_all_nyu40 = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf',
                           'picture', 'counter', 'blinds', 'desk', 'shelves', 'curtain', 'dresser', 'pillow', 'mirror',
                           'floor mat',
                           'clothes', 'ceiling', 'books', 'refrigerator', 'television', 'paper', 'towel',
                           'shower curtain', 'box', 'whiteboard',
                           'person', 'nightstand', 'toilet', 'sink', 'lamp', 'bathtub', 'bag', 'otherstructure',
                           'otherfurniture', 'otherprop']
    sem_ids_all_nyu40 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                         21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]

    sem_names_train_cls19 = ['cabinet', 'bed', 'chair', 'sofa', 'table', 'bookshelf', 'counter', 'desk', 'shelves',
                             'dresser', 'pillow',
                             'refrigerator', 'television', 'box', 'nightstand', 'toilet', 'sink', 'lamp', 'bathtub']
    sem_ids_train_cls19 = [3, 4, 5, 6, 7, 9, 11, 13, 14, 16, 17, 23, 24, 28, 31, 32, 33, 35, 36]

label_map_file = os.path.join(os.path.dirname(__file__), 'scannetv2-labels.combined.tsv')
ins_num_per_img = []

def unzip_raw_2d_files(input_folder, output_folder, scene_names):
    for scene in scene_names:
        print("Unzipping scene: ", scene)
        if os.path.exists(os.path.join(output_folder, scene)):
            print('Deleted and recreate scene folder')
            shutil.rmtree(os.path.join(output_folder, scene))

        ###  extract and save 2D data
        sensor_data_file = os.path.join(input_folder, "scans", scene, scene + '.sens')
        sensor_data = SensorData(sensor_data_file)
        
        scene_f = os.path.join(output_folder, scene)

        # RGB
        rgb_output_folder = os.path.join(scene_f, 'color')
        if not os.path.exists(rgb_output_folder):
            sensor_data.export_color_images(rgb_output_folder)

        # Label
        label_zip_file = os.path.join(input_folder, "scans", scene, scene + '_2d-label-filt.zip')
        os.system("cp {} {}".format(label_zip_file, scene_f))
        os.system("cd {} && unzip {}".format(scene_f, scene + '_2d-label-filt.zip'))
        os.system("rm {}/{}".format(scene_f, scene + '_2d-label-filt.zip'))

        # Instance
        instance_zip_file = os.path.join(input_folder, "scans", scene, scene + '_2d-instance-filt.zip')
        os.system("cp {} {}".format(instance_zip_file, scene_f))
        os.system("cd {} && unzip {}".format(scene_f, scene + '_2d-instance-filt.zip'))
        os.system("rm {}/{}".format(scene_f, scene + '_2d-instance-filt.zip'))
        
        print('Unzip done:', scene)

def represents_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def read_label_mapping(filename, label_from='raw_category', label_to='nyu40id'):
    assert os.path.isfile(filename)
    mapping = dict()
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            mapping[row[label_from]] = int(row[label_to])
    # if ints convert
    if represents_int(list(mapping.keys())[0]):
        mapping = {int(k): v for k, v in mapping.items()}
    return mapping

def map_sem_nyuID(image, label_mapping):
    mapped = np.copy(image)
    keys = np.unique(image)
    for k in keys:
        if k not in label_mapping: continue
        mapped[image == k] = label_mapping[k]
    return mapped

def map_sem_id(image, sem_ids_train):
    mapped_id = np.zeros((image.shape[0], image.shape[1]), dtype=np.int16) - 1
    for sem_i in sem_ids_train:
        new_id = sem_ids_train.index(sem_i)
        mapped_id[image == sem_i] = new_id
    return mapped_id

def map_ins_id(ins_image_in, sem_id):
    ins_image = copy.deepcopy(ins_image_in)
    ins_image[sem_id == -1] = -1  # filter the invalid pixels
    ins_ids = list(set(np.unique(ins_image)) - set([-1]))

    ins_num = len(ins_ids)
    ins_num_per_img.append(ins_num)

    ins_image_new = np.zeros(ins_image.shape, dtype=np.int16) - 1
    for new_id, ins_i in enumerate(ins_ids):
        sem_tp = np.unique(sem_id[ins_image == ins_i])
        if len(sem_tp) > 1:
            print('one ins has more than >1 sem, error');
            exit()
        if sem_tp[0] not in range(len(Data_configs.sem_ids_train_cls19)):
            print('the sem of ins is incorrect');
            exit()
        ins_image_new[ins_image == ins_i] = new_id

    return ins_image_new

def preprocess_imgs(scene_f):
    print('Process folder:', scene_f)
    sem_mapping_dic = read_label_mapping(label_map_file, label_from='id', label_to='nyu40id')

    out_sem_f_id = scene_f + '/label-filt-cls' + str(len(Data_configs.sem_ids_train_cls19)) + '/'
    if os.path.exists(out_sem_f_id): print('deleted and recreate sem id'); shutil.rmtree(out_sem_f_id)
    os.makedirs(out_sem_f_id)

    out_ins_f_id = scene_f + '/instance-filt-cls' + str(len(Data_configs.sem_ids_train_cls19)) + '/'
    if os.path.exists(out_ins_f_id): print('deleted and recreate ins id'); shutil.rmtree(out_ins_f_id)
    os.makedirs(out_ins_f_id)

    total_imgs = sorted(glob.glob(scene_f + '/color/*.jpg'))
    for i in range(len(total_imgs)):
        sem_f = scene_f + '/label-filt/' + str(i) + '.png'
        ins_f = scene_f + '/instance-filt/' + str(i) + '.png'

        ## proj sem
        sem_2d_label_rawID = np.asarray(imageio.imread(sem_f), dtype=np.int16)
        sem_2d_label_nyuID = map_sem_nyuID(sem_2d_label_rawID, sem_mapping_dic)
        sem_2d_label_id = map_sem_id(sem_2d_label_nyuID, Data_configs.sem_ids_train_cls19)
        np.savez_compressed(out_sem_f_id + str(i) + '.npz', sem_2d_label_id=sem_2d_label_id)

        ## proj ins
        ins_2d_label_rawID = np.asarray(imageio.imread(ins_f), dtype=np.int16)
        ins_2d_label_id = map_ins_id(ins_2d_label_rawID, sem_2d_label_id)
        np.savez_compressed(out_ins_f_id + str(i) + '.npz', ins_2d_label_id=ins_2d_label_id)

    return 0
