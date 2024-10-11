## Prerequisites 
* python 3.8.8
* [pytorch](https://pytorch.org/) (tested on Release 1.8.1)

## Installation
* git clone https://github.com/xiumin123/Face_inpainting.git
* cd Face_inpainting
* conda env create -f environment.yml 
* conda activate inpainting

## Getting Started
* prepare dataset
* train:
  ** cd src 
  ** python train.py
* test
  ** cd src 
  ** python test.py --pre_train [path to pretrained model]
* Evaluating
  ** cd src 
  ** python eval.py --real_dir [ground truths] --fake_dir [inpainting results] --metric mae psnr ssim fid
