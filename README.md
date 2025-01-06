## Semantic-Guided Face Inpainting with Subspace Pyramid Aggregation
![image](https://github.com/xiumin123/Face_inpainting/blob/main/data/%E5%9B%BE%E7%89%871.png)

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
* download pretrained model,place this model under the floder src/model/model_pr
  
  https://pan.baidu.com/s/1pHgGfYht8vKq1tJyEpE6Iw 提取码: 7r9i 
* train:
   * Before you train the model, you can choose the model variants in the "src/model" folder. You can modify the necessary parameters in "src/utils/options.py".  
   * cd src 
   * python train.py
* test:
   * cd src 
   * python test.py --pre_train [path to pretrained model]
* Evaluating:
   * cd src 
   * python eval.py --real_dir [ground truths] --fake_dir [inpainting results] --metric mae psnr ssim fid

## Acknowledgement
  Our models were trained and tested on an RXT3090Ti GPU and Intel(R) Xeon(R) Silver 4214 CPU @ 2.20GHz.All the experimental data in this thesis were reproduced by myself on the server.Further more, our experiment is developed relying on [AOT-GAN](https://github.com/researchmm/AOT-GAN-for-Inpainting) and other projects. Thanks for these great projects.
