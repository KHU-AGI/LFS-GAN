# [ICCV 2023] LFS-GAN: Lifelong Few-Shot Image Generation

Official PyTorch implementation for ICCV 2023 paper:

**LFS-GAN: Lifelong Few-Shot Image Generation**  
Juwon Seo*, Ji-Su Kang*, and Gyeong-Moon Park<sup>✝︎</sup> 

[![arXiv](https://img.shields.io/badge/arXiv-2308.11917-b31b1b.svg)](https://arxiv.org/abs/2308.11917)  

# Environment
- Python 3.10.8
- PyTorch 1.12.0
- Torchvision 0.13.0
- NVIDIA GeForce RTX 3090


# Preparation
## Environment
Before running our code, please set up an environment by running commands below:
```bash
git clone git@github.com:JJuOn/LFS-GAN.git
cd LFS-GAN
conda env create -f environmenmt.yaml
```
## Dataset
We are providing the training datasets (10-shot) and their full dataset for evaluation.  
We recommend you to extract these training sets to `./data` directory.

| |Sketches|Female|Sunglasses|Male|Babies| 
|--|--|--|--|--|--|
|10-shot|[Download]()|[Download]()|[Download]()|[Download]()|[Download]()|
|Full|[Download]()|[Download]()|[Download]()|[Download]()|[Download]()|

Before running `train.py`, please process training datasets to be lmdb.

```bash
python prepare_data.py --out processed_data/{dataset} --size 256 data/{dataset}
```
For example, if you want process `Sketches` dataset, run the command below:
```bash
python prepare_data.py --out processed_data/Sketches --size 256 data/Sketches
```

## Pretrained Checkpoints
Here, we provide a checkpoint of StyleGAN2 pretrained on FFHQ-256.  
You can download the model checkpoint via google drive below.
||FFHQ-256|
|--|--|
|Pretrained StyleGAN2| [Download]()

Since our code is based on [@rosinaltiy's implementation](https://github.com/rosinality/stylegan2-pytorch), the checkpoint is only compatibile with this implementation.

We also provide checkpoints of our framework - LFS-GAN on Sketches, Female, Sunglasses, Male, and Babies.

| |Sketches|Female|Sunglasses|Male|Babies| 
|--|--|--|--|--|--|
|Pretrained LFS-GAN|[Download]()|[Download]()|[Download]()|[Download]()|[Download]()|

# Train
You can train LFS-GAN by running:
```bash
python train.py --data_path processed_data/{dataset} --ckpt ffhq.pt --exp lfs-gan \
                --rank 1 --left_use_add --left_use_act --cluster_wise_mode_seeking
```

The trained checkpoints are saved to `./checkpoints`

# Evaluation
Before the evaluation of the trained model, you first sample images:
```bash
python generate.py --pretrained_ckpt ffhq.pt \
                   --ckpt checkpoints/lfs-gan/{some_checkpoint_name}.pt \
                   --result_path fake_images/lfs-gan/{dataset} \
                   --rank 1 --left_use_act --left_use_add
```
We also provide the generated images from our pretrained LFS-GAN.  
We have samples 5,000 images per task.

| |Sketches|Female|Sunglasses|Male|Babies| 
|--|--|--|--|--|--|
|Fake images|[Download]()|[Download]()|[Download]()|[Download]()|[Download]()|

You can measure the generation quality by using `pytorch-fid`.
```
python -m pytorch_fid {real_path} fake_images/lfs-gan/{dataset} --device cuda
```
The `{real_path}` denotes the path of the full dataset.

You can also measure the generation diversity by running `evaluate_b_lpips.py`.

```
python evaluate_b_lpips.py --real_path processed_data/{dataset} --fake_path fake_images/lfs-gan/{dataset}
```

# Acknowledgment
This code is based on [stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch), [few-shot-gan-adaptation](https://github.com/utkarshojha/few-shot-gan-adaptation), and [PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity).

# BibTex
```
@inproceedings{seo2023lfs
  title    = {LFS-GAN: Lifelong Few-Shot Image Generation}
  author   = {Seo, Juwon and Kang, Ji-Su and Park, Gyeong-Moon}
  booktitle= {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)}
  year     = {2023}
}
```
