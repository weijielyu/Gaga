# *Gaga* : Group Any Gaussians via 3D-aware Memory Bank

## Installation
```
conda create -n gaga python=3.9 -y
conda activate gaga
conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch
pip install plyfile tqdm scipy wandb opencv-python scikit-learn lpips torchmetrics imageio
```

- Use Segment Anything (SAM).
```
pip install git+https://github.com/facebookresearch/segment-anything.git
```
 Download model checkpoint [here](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints).

- Use EntitySeg.
```
cd Gaga/mask
# Install detectron 2 (https://detectron2.readthedocs.io/en/latest/tutorials/install.html)
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
# Install EntitySeg
git clone https://github.com/qqlu/Entity.git
cp -r Entity/Entityv2/CropFormer/ detectron2/detectron2/projects/
cd detectron2/detectron2/projects/CropFormer/entity_api/PythonAPI
make
```

Download model checkpoint [here](https://huggingface.co/datasets/qqlu1992/Adobe_EntitySeg/tree/main/CropFormer_model/Entity_Segmentation/CropFormer_hornet_3x).

You can also refer to [EntitySeg Github Repo](https://github.com/qqlu/Entity/blob/main/Entityv2/CODE.md) for usage.

After downloading the model ckpts, please modify the paths in `mask/config.json` to accommodate your setting.

## Datasets
### MipNeRF 360
Download the MipNeRF 360 dataset from their [project website](https://jonbarron.info/mipnerf360/).
```
wget http://storage.googleapis.com/gresearch/refraw360/360_v2.zip
unzip 360_v2.zip -d {your_dataset_path}/mipnerf360
```
The MipNeRF 360 dataset will be in `{your_dataset_path}/mipnerf360`.

### Replica
We use pre-rendered Replica dataset provided by [Semantic-NeRF](https://github.com/Harry-Zhi/semantic_nerf/tree/main).

Dropbox Link: `https://www.dropbox.com/sh/9yu1elddll00sdl/AAC-rSJdLX0C6HhKXGKMOIija?dl=0`.

After downloading, unzip file with:
```
sh data/replica/unzip_replica.sh {zip_file_path.zip} {unzip_file_path}
```

Then process the Replica dataset with:
```
python data/process.py \
    --input_folder {unzip_file_path}/Replica_Dataset \
    --dataset_folder {your_dataset_path} \
    --dataset replica
```
The processed Replica dataset will be in `{your_dataset_path}/replica`.

### ScanNet
Download ScanNet dataset here: `http://www.scan-net.org/`.

After downloading, process the ScanNet dataset with:
```
python data/process.py \
    --input_folder {scannet_folder_path} \
    --dataset_folder {your_dataset_path} \
    --dataset scannet
```
The processed ScanNet dataset will be in `{your_dataset_path}/scannet`.

### LERF-Mask

Please refer to [Gaussian Grouping](https://github.com/lkeab/gaussian-grouping/blob/main/docs/dataset.md) for using LERF-Mask dataset.

## Usage
We organize the related folders as follows:
```
Gaga
|-- dataset ({your_dataset_path})
    |-- mipnerf360
        |-- bicycle
        |-- ...
    |-- replica
        |-- office_0
        |-- ...
    |-- ...
|-- weight ({your_weight_path})
    |-- sam_vit_h_4b8939.pth
    |-- CropFormer_hornet_3x_03823a.pth
|-- model ({your_model_path})  # Pre-trained vanilla 3DGS
    |-- mipnerf360
        |-- bicycle
        |-- ...
    |-- replica
        |-- office_0
        |-- ...
    |-- ...
|-- Output ({your_output_path})  # Segmentation-awared 3DGS
    |-- mipnerf360
        |-- bicycle
        |-- ...
    |-- replica
        |-- office_0
        |-- ...
    |-- ...
```

### Step 1: Train Vanilla Gaussian Splatting
If you have pre-trained 3D Gaussians, place it in `{your_model_path}`.

Otherwise, please refer to [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) to train 3D Gaussians.

### Step 2: Get 2D Class-agnostic Masks

**SAM**
```
python mask/get_raw_mask.py \
    --dataset_folder {your_dataset_path} \
    --scene replica/office_0 \  # For example
    --image images \
    --seg_method sam \
    --visualize
```

**EntitySeg**
```
python mask/get_raw_mask.py \
    --dataset_folder {your_dataset_path} \
    --scene replica/office_0 \  # For example
    --image images \
    --seg_method entityseg \
    --visualize
```

### Step 3: Mask Association
```
python associate.py \
    --source_path {your_dataset_path}/replica/office_0 \
    --model_path {your_model_path}/replica/office_0 \
    --seg_method sam \
    --visualize
```

### Step 4: Lift Masks to 3D
```
python lift.py \
    --source_path {your_dataset_path}/replica/office_0 \
    --model_path {your_output_path}/replica/office_0/sam \
    --trained_model_path {your_model_path}/replica/office_0 \
    --object_path sam_mask    \
    --eval

python render.py -m {your_output_path}/replica/office_0/sam
```

### Step 5 (Optional): Evaluaion on Replica and ScanNet Datasets
```
python eval.py   \
    --gt_masks {your_dataset_path}/replica/office_0/semantic_instance  \
    --pred_masks {your_output_path}/replica/office_0/sam/test/ours_10000/objects_test
```

## Acknowledgement

Our codes are based on the following GitHub repos. Thanks for their wonderful implementations!
```
https://github.com/graphdeco-inria/gaussian-splatting
https://github.com/lkeab/gaussian-grouping
https://github.com/VITA-Group/FSGS
https://github.com/Harry-Zhi/semantic_nerf
https://github.com/vLAR-group/DM-NeRF
```
