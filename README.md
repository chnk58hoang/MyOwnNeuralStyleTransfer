<div align="center">


<h1> Neural Style Transfer

[![python](https://img.shields.io/badge/-Python_3.8-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![PyTorch](https://img.shields.io/badge/-PyTorch_2.0-white?logo=pytorch&logoColor=orange)](https://pytorch.org/)

</h1>

</div>

## Introduction

This is a PyTorch implementation of [Image Style Transfer Using Convolutional Neural Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf).

## Installation
I highly recommend you to use a conda virtual environment.
```bash
conda create -n nst python=3.8
pip install -r requirements.txt
```

## Usage

```python
python  src/transfer.py --content_img [path to the content image] --style_img [path to the style image] --save_path [path to save the generated image] --steps [number of transfer steps]
```

