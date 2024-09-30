# DiMR: Alleviating Distortion in Image Generation via Multi-Resolution Diffusion Models (NeurIPS 2024)
[[project page](https://qihao067.github.io/projects/DiMR)] | [[paper](https://arxiv.org/pdf/2406.09416)] | [[arxiv](https://arxiv.org/abs/2406.09416)]

______

![DiMR](https://github.com/qihao067/DiMR/blob/main/imgs/DiMR.jpeg)

We propose DiMR, a new diffusion backbone that achieves state-of-the-art image generation. For example, on the ImageNet 256 x 256 benchmark, **DiMR, with only 505M parameters, surpasses all existing image generation models of various sizes**, without any bells and whistles.

![DiMR](https://github.com/qihao067/DiMR/blob/main/imgs/256_all.jpeg)

In addition, with the proposed Multi-Resolution Network, DiMR alleviates distortions and enhances visual fidelity without increasing computational costs.

![DiMR](https://github.com/qihao067/DiMR/blob/main/imgs/DiMR-distortion_projectpage.jpg)

______

## Requirements

The code has been tested with PyTorch 2.1.2 and Cuda 12.1.

An example of installation commands is provided as follows:

```
git clone https://github.com/qihao067/DiMR.git
cd DiMR

## environment setup
pip3 install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip3 install accelerate==0.12.0 absl-py ml_collections einops wandb ftfy==6.1.1 transformers==4.23.1

pip3 install -U --pre triton
pip3 install timm
pip3 install tensorboard
```

______

## Pretrained Models

| Model | Image Resolutions | Training Epochs | #Params. | Gflops | FID50K | Download |
| :---------------- | ----------------- | --------------- | -------- | ------ | ------ | -------- |
| DiMR-XL/2R | 256x256           | 800             | 505M     | 160    | 1.70   | [[Link](https://huggingface.co/QHL067/DiMR/blob/main/DiMR-XL_2R_800epochs.pth)] |
| DiMR-G/2R  | 256x256           | 800             | 1.06B    | 331    | 1.63   | [[Link](https://huggingface.co/QHL067/DiMR/blob/main/DiMR-G_2R_800epochs.pth)] |

Please note that these models are trained only on limited academic dataset ImageNet, and they are only for research purposes.

______

## Data Preparation

The data preparation protocol strictly follows [U-ViT](https://github.com/baofff/U-ViT?tab=readme-ov-file#preparation-before-training-and-evaluation). Many thanks to the authors for their outstanding efforts.

**Step 1: Download the auto-encoder from Stable Diffusion:**

Download the `stable-diffusion` directory from this [link](https://drive.google.com/drive/folders/1yo-XhqbPue3rp5P57j6QbA5QZx6KybvP?usp=sharing) (which contains image autoencoders converted from [Stable Diffusion](https://github.com/CompVis/stable-diffusion)). Place the downloaded directory in this codebase as `assets/stable-diffusion`. The autoencoders are used in latent diffusion models.

**Step 2: Prepare ImageNet:**

Download the original ImageNet dataset and extract its features using `scripts/extract_imagenet_feature.py`. Make sure you also update the path of the extracted features in the config file (`configs/DiMR-G-2R_imagenet256.py` or `configs/DiMR-XL-2R_imagenet256.py`).

**Step 3: Prepare reference statistics for FID**

Download the `fid_stats` directory from this [link](https://drive.google.com/drive/folders/1yo-XhqbPue3rp5P57j6QbA5QZx6KybvP?usp=sharing) (which contains reference statistics for FID). Place the downloaded directory in this codebase as `assets/fid_stats`. These reference statistics are used to monitor FID during the training process, in addition to evaluation.

______

## Training DiMR

We provide a training script for training class-conditional DiMR models on ImageNet 256 x 256 from scratch. It can be easily modified to support different resolutions and datasets. To reproduce the results, launch the training with 16 GPUs on 2 nodes using the following commands:

```
accelerate launch --multi_gpu --num_processes 16 --num_machines 2 --mixed_precision fp16 train.py \
       --config=configs/DiMR-XL-2R_imagenet256.py
```

______

## Evaluation

Following previous methods, we use [ADM's TensorFlow evaluation suite](https://github.com/openai/guided-diffusion/tree/main/evaluations) to compute FID, Inception Score, and other metrics. To do so, you first need to sample 50K images from our pre-trained DiMR model using `N` GPUs:

```
rm -rf saved_images* class_lab.txt

accelerate launch --multi_gpu --num_processes N --mixed_precision fp16 eval.py \
       --config=configs/DiMR-XL-2R_imagenet256.py \
       --nnet_path='path/to/the/checkpoint' \
       --IMGsave_path=saved_images
```

The generated images will be saves in `saved_images`, and the class labels will be saved in `class_lab.txt`.

Then, run the following script to convert the generated images into a `.npz` file:

```
python3 img2npz.py
```

After that, please follow [ADM's TensorFlow evaluation suite](https://github.com/openai/guided-diffusion/tree/main/evaluations) to compute all metrics.

In addition, following [U-ViT](https://github.com/baofff/U-ViT), we will also report an FID score computed by a [PyTorch implementation](https://github.com/mseitzer/pytorch-fid) when sampling 50K images using `eval.py`. However, this is only used to help monitor training. For a fair comparison with DiT, we report the results computed by [ADM's TensorFlow evaluation suite](https://github.com/openai/guided-diffusion/tree/main/evaluations).

______

## Terms of use

The project is created for research purposes.

______

## Acknowledgements

This codebase is built upon the following repository:

- [[U-ViT](https://github.com/baofff/U-ViT)]

Much appreciation for their outstanding efforts.

____________

## License

The code in this repository is released under the Apache License, Version 2.0.

______

## BibTeX

If you use our work in your research, please use the following BibTeX entry.

```
@article{liu2024alleviating,
  title={Alleviating Distortion in Image Generation via Multi-Resolution Diffusion Models},
  author={Liu, Qihao and Zeng, Zhanpeng and He, Ju and Yu, Qihang and Shen, Xiaohui and Chen, Liang-Chieh},
  journal={arXiv preprint arXiv:2406.09416},
  year={2024}
}
```

