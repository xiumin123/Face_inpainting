import argparse

parser = argparse.ArgumentParser(description='Image Inpainting')

# data specifications 
parser.add_argument('--dir_image', type=str, default="../datasets/CelebA-HQ-img-256-resize",#../dataset   ../../FFHQ
                    help='image dataset directory')
parser.add_argument('--dir_mask', type=str, default="../datasets/CelebA-HQ-img-256-resize",#../dataset
                    help='mask dataset directory')
parser.add_argument('--data_train', type=str, default='train1000',  #train
                    help='dataname used for training')
parser.add_argument('--data_test', type=str, default='test_',#ffhq_256/testjpg #test
                    help='dataname used for testing')
parser.add_argument('--image_size', type=int, default=256,
                    help='image size used during training')
parser.add_argument('--mask_type', type=str, default='pconv_0',#pconv_
                    help='mask used during training')
parser.add_argument('--dir_test', type=str, default="../datasets/CelebA-HQ-img-256-resize/test",#../datasets/CelebA-HQ-img-256-resize/test   ../../FFHQ/ffhq_256/testjpg/testjpg1
                    help='test dataset directory')
# model specifications 
parser.add_argument('--model', type=str, default='aotgan',
                    help='model name')
parser.add_argument('--block_num', type=int, default=8,
                    help='number of AOT blocks')
parser.add_argument('--rates', type=str, default='1+2+4+8',
                    help='dilation rates used in AOT block')
parser.add_argument('--gan_type', type=str, default='smgan',
                    help='discriminator types')

# hardware specifications 
parser.add_argument('--seed', type=int, default=2021,
                    help='random seed')
parser.add_argument('--num_workers', type=int, default=4,
                    help='number of workers used in data loader')

# optimization specifications 
parser.add_argument('--lrg', type=float, default=1e-4,
                    help='learning rate for generator')
parser.add_argument('--lrd', type=float, default=1e-4,
                    help='learning rate for discriminator')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--beta1', type=float, default=0.5,
                    help='beta1 in optimizer')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='beta2 in optimier')

# loss specifications 
parser.add_argument('--rec_loss', type=str, default='1*L1+250*Style+0.1*Perceptual',
                    help='losses for reconstruction')
parser.add_argument('--adv_weight', type=float, default=0.01,
                    help='loss weight for adversarial loss')

# training specifications 
parser.add_argument('--iterations', type=int, default=100000,  #1e6
                    help='the number of iterations for training')
parser.add_argument('--batch_size', type=int, default=1,  #8
                    help='batch size in each mini-batch')
parser.add_argument('--port', type=int, default=22334,
                    help='tcp port for distributed training')
parser.add_argument('--resume', action='store_true',
                    help='resume from previous iteration')


# log specifications 
parser.add_argument('--print_every', type=int, default=1000,
                    help='frequency for updating progress bar')
parser.add_argument('--save_every', type=int, default=10000,
                    help='frequency for saving models')#1e4
parser.add_argument('--save_dir', type=str, default='../experiments',
                    help='directory for saving models and logs')
parser.add_argument('--tensorboard', action='store_true',
                    help='default: false, since it will slow training. use it for debugging')

# test and demo specifications 
parser.add_argument('--pre_train', type=str, default="../experiments/aotgan_train1000_pconv_0256_pconv_1000_100000/G0010000.pt", 
#/aotgan_train1000_pconv_256_pconv_1000_100000：小遮挡
                    help='path to pretrained models')
parser.add_argument('--outputs', type=str, default='../1000_100000',  #outputs
                    help='path to save results')
parser.add_argument('--thick',  type=int, default=15, 
                    help='the thick of pen for free-form drawing')
parser.add_argument('--painter', default='freeform', choices=('freeform', 'bbox'),
                    help='different painters for demo ')


# ----------------------------------
args = parser.parse_args()
args.iterations = int(args.iterations)

args.rates = list(map(int, list(args.rates.split('+'))))

losses = list(args.rec_loss.split('+'))
args.rec_loss = {}
for l in losses: 
    weight, name = l.split('*')
    args.rec_loss[name] = float(weight)