# [ICCV 2023] LFS-GAN: Lifelong Few-Shot Image Generation

Official PyTorch implementation for ICCV 2023 paper:

**LFS-GAN: Lifelong Few-Shot Image Generation**  
Juwon Seo*, Ji-Su Kang*, and Gyeong-Moon Park<sup> $\dagger$ </sup> 

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
|10-shot|[Download](https://drive.google.com/file/d/1QvvPiY0Br7bS5eFOjAw7fWrDvCdnDeRE/view?usp=drive_link)|[Download](https://drive.google.com/file/d/10C9aBzRF4GW68URfUcm1bw_R5xHr7B-6/view?usp=drive_link)|[Download](https://drive.google.com/file/d/1OWJMQC1RhEkwX9UwJAefVNwHS23EOA15/view?usp=drive_link)|[Download](https://drive.google.com/file/d/1DjlEcs6_W2cg26lbWdyTlsRQHeFTF0xg/view?usp=drive_link)|[Download](https://drive.google.com/file/d/13Y6sdqUx75xJTCZ0f5MYV7IvecMFp3CT/view?usp=drive_link)|
|Full|[Download](https://drive.google.com/file/d/1aM9fe7LUQelLIc09FLUdEy-wdlyJ_boK/view?usp=drive_link)|[Download](https://drive.google.com/file/d/11r6dlaQioXWSwF4Evo7RKt5cfYSIBehP/view?usp=drive_link)|[Download](https://drive.google.com/file/d/19NkyLLI87v92vL_3KqE7EJUZ0bq9ZY0n/view?usp=drive_link)|[Download](https://drive.google.com/file/d/1g-4B5IOvTeGyM6W3655OsFL-o9L5kka_/view?usp=drive_link)|[Download](https://drive.google.com/file/d/1H48T5fdZoqwlQXAmaaI5nSqMDwpeUWgf/view?usp=drive_link)|

Before running `train.py`, please process training datasets to be lmdb.

```bash
python prepare_data.py --out processed_data/{dataset} --size 256 data/{dataset}
```
For example, if you want to process `Sketches` dataset, run the command below:
```bash
python prepare_data.py --out processed_data/Sketches --size 256 data/Sketches
```

## Pretrained Checkpoints
Here, we provide a checkpoint of StyleGAN2 pretrained on FFHQ-256.  
You can download the model checkpoint via google drive below.
||FFHQ-256|
|--|--|
|Pretrained StyleGAN2| [Download](https://drive.google.com/file/d/1TQ_6x74RPQf03mSjtqUijM4MZEMyn7HI/view?usp=sharing)

Since our code is based on [@rosinaltiy's implementation](https://github.com/rosinality/stylegan2-pytorch), the checkpoint is only compatibile with this implementation.

We also provide checkpoints of our framework - LFS-GAN on Sketches, Female, Sunglasses, Male, and Babies.

| |Sketches|Female|Sunglasses|Male|Babies| 
|--|--|--|--|--|--|
|Pretrained LFS-GAN|[Download](https://drive.google.com/file/d/11zUgRUeb8dvSHYlxMxqxTaxso-js56ds/view?usp=drive_link)|[Download](https://drive.google.com/file/d/1sFlUoddQRgKJ0WeI6PFNJkh4rQ0JDr36/view?usp=drive_link)|[Download](https://drive.google.com/file/d/16U6ajA9Pk9p8IDW57euHsHJ0HqW3IcIa/view?usp=drive_link)|[Download](https://drive.google.com/file/d/1qKYFWbOvYebWWjUxYoaHGHdx7L1Fmk4y/view?usp=drive_link)|[Download](https://drive.google.com/file/d/11r0PefSWCvYkm2VEy0QuVz0d6nag1xNs/view?usp=drive_link)|

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
|Fake images|[Download](https://drive.google.com/file/d/1m33hhIWJw40eaJSobopdFc7JK1DNe04p/view?usp=sharing)|[Download](https://drive.google.com/file/d/1jvG3avygJ7_Vp2pl3RYmFJcCwsKSgHuC/view?usp=drive_link)|[Download](https://drive.google.com/file/d/1e3TCo1ykqAOJ-7E3tFlRBUFRuSJZhrEg/view?usp=drive_link)|[Download](https://drive.google.com/file/d/13auoqCpulPy0YNsNEP4g6gqK7n4GFOvN/view?usp=drive_link)|[Download](https://drive.google.com/file/d/1kGJJLIvt6_Cj_Anf5C9fPMZ5hbA3QGRR/view?usp=drive_link)|

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
This code is based on [stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch), [few-shot-gan-adaptation](https://github.com/utkarshojha/few-shot-gan-adaptation), [CelebAHQ-Gender](https://github.com/JJuOn/CelebAHQ-Gender), and [PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity).

# BibTex
```
@inproceedings{seo2023lfs
  title    = {LFS-GAN: Lifelong Few-Shot Image Generation}
  author   = {Seo, Juwon and Kang, Ji-Su and Park, Gyeong-Moon}
  booktitle= {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)}
  year     = {2023}
}
```
