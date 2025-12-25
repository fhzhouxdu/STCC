# Project Overview

This repository is the official repository of paper "Description Helps: Semantic and Texture Consistency Constraints for SAR-to-Optical Translation"


----

## Environment Dependencies

- Recommended conda environment:

```bash
conda create -n STCC python=3.10 -f ./environment.yaml
conda activate STCC
```

----

## Downloading Stable Diffusion Weights

You need to decide which Stable Diffusion Model you want to control. In this example, we will just use standard SD1.5. You can download it from the official page of Stability. You want the file "[v1-5-pruned.ckpt](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main)".

(Or "[v2-1_512-ema-pruned.ckpt](https://huggingface.co/Manojb/stable-diffusion-2-1-base/tree/main)" if you are using SD2.)

----

## ControlNet config

Note that all weights inside the ControlNet are also copied from SD so that no layer is trained from scratch, and you are still finetuning the entire model.
We provide a simple script for you to achieve this easily. If your SD filename is "./models/v1-5-pruned.ckpt" and you want the script to save the processed model (SD+ControlNet) at location "./models/control_sd15_ini.ckpt", you can just run:
```bash
python tool_add_control.py ./models/v1-5-pruned.ckpt ./models/control_sd15_ini.ckpt
```
Or if you are using SD2:
```bash
python tool_add_control_sd21.py ./models/v2-1_512-ema-pruned.ckpt ./models/control_sd21_ini.ckpt
```

----

## Dataset Preparation and Format

Example dataset layout:
```
sen1-2/
  sen1-2_train_rs-llava.json
  sen1-2_test_rs-llava.json
  trainA/
    img_0001.jpg
    img_0002.jpg
  trainB/
    img_0001.jpg
    img_0002.jpg
  testA/
    img_0003.jpg
    img_0004.jpg
  testB/
    img_0003.jpg
    img_0004.jpg
```

You can also change data path in `./tutorial_dataset.py` by yourself.

----

## Training and Fine-tuning

- Train
```bash
python tutorial_train_sd21.py
```

- Finetune
```bash
python finetune.py
```

----

## Inference
```bash
python test.py
```

----

## Baidu Netdisk (Datasets) — Placeholder

- Link: [https://pan.baidu.com/s/1t3Vog9C_JFrG5d1jrqEzmA?pwd=77g2]`
- Extraction code: `77g2`

----
