# CMUNeXt: An Efficient Medical Image Segmentation Network based on Large Kernel and Skip Fusion

Official Pytorch Code base for "CMUNeXt: An Efficient Medical Image Segmentation Network based on Large Kernel and Skip Fusion"

[Paper](https://arxiv.org/pdf/2308.01239.pdf) | [Code](https://github.com/FengheTan9/CMUNeXt)

**News** 🥰:

- <font color="#dd0000" size="4">**Paper is accepted by ISBI 2024 !**</font> 🎉
- <font color="#dd0000" size="4">**Code is released now !**</font> 
- <font color="#dd0000" size="4">**Code will be released soon !**</font> 😘

## Introduction
The U-shaped architecture has emerged as a crucial paradigm in the design of medical image segmentation networks. However, due to the inherent local limitations of convolution, a fully convolutional segmentation network with U-shaped architecture struggles to effectively extract global context information, which is vital for the precise localization of lesions. While hybrid architectures combining CNNs and Transformers can address these issues, their application in real medical scenarios is limited due to the computational resource constraints imposed by the environment and edge devices. In addition, the convolutional inductive bias in lightweight networks adeptly fits the scarce medical data, which is lacking in the Transformer based network. In order to extract global context information while taking advantage of the inductive bias, we propose CMUNeXt, an efficient fully convolutional lightweight medical image segmentation network, which enables fast and accurate auxiliary diagnosis in real scene scenarios. CMUNeXt leverages large kernel and inverted bottleneck design to thoroughly mix distant spatial and location information, efficiently extracting global context information. We also introduce the Skip-Fusion block, designed to enable smooth skip-connections and ensure ample feature fusion. Experimental results on multiple medical image datasets demonstrate that CMUNeXt outperforms existing heavyweight and lightweight medical image segmentation networks in terms of segmentation performance, while offering a faster inference speed, lighter weights, and a reduced computational cost.

### CMUNeXt:

![framework](imgs/CMUNeXt.png)

## Performance Comparison

<img src="imgs/GFLOPs.png" width="260"> <img src="imgs/Params.png" width="260">  <img src="imgs/FPS.png" width="260">   

## Datasets

Please put the [BUSI](https://www.kaggle.com/aryashah2k/breast-ultrasound-images-dataset) dataset or your own dataset as the following architecture. 
```
└── CMUNeXt
    ├── data
        ├── busi
            ├── images
            |   ├── benign (10).png
            │   ├── malignant (17).png
            │   ├── ...
            |
            └── masks
                ├── 0
                |   ├── benign (10).png
                |   ├── malignant (17).png
                |   ├── ...
        ├── your dataset
            ├── images
            |   ├── 0a7e06.png
            │   ├── ...
            |
            └── masks
                ├── 0
                |   ├── 0a7e06.png
                |   ├── ...
    ├── dataloader
    ├── network
    ├── utils
    ├── main.py
    └── split.py
```

``` bash
# source destination
ln -s /home/xr/DEV/Data/arm data/arm 
```
## Environment

- GPU: NVIDIA GeForce RTX4090 GPU
- Pytorch: 1.13.0 cuda 11.7
- cudatoolkit: 11.7.1
- scikit-learn: 1.0.2

```bash
conda create -n cmunext python=3.10 -y
conda activate cmunext
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Training and Validation

You can first split your dataset:

```python
# python split.py --dataset_name wrist --dataset_root ./data --img_ext .jpg
# python split.py --dataset_name arm --dataset_root ./data --img_ext .jpg
python split.py --dataset_name busi --dataset_root ./data --img_ext .png
```

Then, train and validate your dataset:

```python
# python main.py --model CMUNeXt-L --base_dir ./data/wrist --train_file_dir wrist_train.txt --val_file_dir wrist_val.txt --img_ext .jpg --num_classes 1
# python main.py --model CMUNeXt-L --base_dir ./data/busi --train_file_dir busi_train.txt --val_file_dir busi_val.txt --img_ext .png  --num_classes 2
# python main.py --model CMUNeXt-L --base_dir ./data/arm --train_file_dir arm_train.txt --val_file_dir arm_val.txt --img_ext .jpg  --num_classes 2
python main.py --model ["CMUNeXt", "CMUNeXt-S", "CMUNeXt-L"] --base_dir ./data/busi --train_file_dir busi_train.txt --val_file_dir busi_val.txt
# DDP
torchrun --standalone --nnodes=1 --nproc_per_node=1 main_ddp.py --model CMUNeXt-L --base_dir ./data/arm --train_file_dir arm_train.txt --val_file_dir arm_val.txt --img_ext .jpg  --num_classes 17 --batch_size 32
```

## Acknowledgements:

This code-base uses helper functions from [Medical-Image-Segmentation-Benchmarks](https://github.com/FengheTan9/Medical-Image-Segmentation-Benchmarks).

## Citation

If you use our code, please cite our paper:

```tex
@article{tang2023cmunext,
  title={CMUNeXt: An Efficient Medical Image Segmentation Network based on Large Kernel and Skip Fusion},
  author={Tang, Fenghe and Ding, Jianrui and Wang, Lingtao and Ning, Chunping and Zhou, S Kevin},
  journal={arXiv preprint arXiv:2308.01239},
  year={2023}
}
```

