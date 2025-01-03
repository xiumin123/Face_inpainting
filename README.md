##  prepare datasets

 * Download public image datasets or your own image datasets and download the  public masks.
 * Split your dataset into training set, test set and validation set according to the ratio of 8:1:1.
 * Specify the path to training data by --dir_image and --dir_mask.
  
## Requirements
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
   * cd src 
   * python train.py
* test:
   * cd src 
   * python test.py --pre_train [path to pretrained model]
* Evaluating:
   * cd src 
   * python eval.py --real_dir [ground truths] --fake_dir [inpainting results] --metric mae psnr ssim fid
