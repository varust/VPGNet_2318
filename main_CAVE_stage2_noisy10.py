from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.io as io
import os
import random
import time
import socket

from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torch.utils.data import DataLoader
from network_stage2 import DGSMPIR
from data import get_patch_training_set, get_test_set
from torch.autograd import Variable
from psnr import MPSNR
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
# from apex import amp
# from apex.parallel import DistributedDataParallel
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.distributed.autograd as dist_autograd
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.optim import DistributedOptimizer
from torch.distributed.rpc import RRef
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
os.environ["CUDA_VISIBLE_DEVICES"] = "3"



# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--nEpochs', type=int, default=105, help='number of epochs to train for')
parser.add_argument('--mode', default=11, type=int, help='Train or Test.')

parser.add_argument('--save_folder', default='TrainedNet_CAVE_stage2_noisy10/', help='Directory to keep training outputs.')
parser.add_argument('--outputpath', type=str, default='result_CAVE_stage2_noisy10/', help='Path to output img')
parser.add_argument('--batchSize', type=int, default=7, help='training batch size')

parser.add_argument('--upscale_factor', type=int, default=8, help="super resolution upscale factor")
parser.add_argument('--patch_size', type=int, default=64, help='training patch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--ChDim', type=int, default=31, help='output channel number')
# parser.add_argument('--alpha', type=float, default=0.2, help='alpha')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate. Default=0.01')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')

parser.add_argument('--local_rank', default=0, type=int, help='None')
parser.add_argument('--use-distribute', type=bool, default=False, help='None')
opt = parser.parse_args()

print(opt)

# if opt.cuda and not torch.cuda.is_available():
#     raise Exception("No GPU found, please run without --cuda")


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_random_seed(opt.seed)

use_dist = opt.use_distribute
if use_dist:
    # torch.cuda.set_device(rank % torch.cuda.device_count())
    dist.init_process_group(backend="nccl", init_method='env://')

print('===> Loading datasets')
train_set = get_patch_training_set(opt.upscale_factor, opt.patch_size)
if use_dist:
    sampler = DistributedSampler(train_set)
test_set = get_test_set()

if use_dist:
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, sampler = sampler, pin_memory=True)
else:
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle = True, pin_memory=True)
    testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False, pin_memory=True)

print('===> Building model')


print("===> distribute model")


# data = io.loadmat('./P.mat')
# P    = data['P']
# P    = torch.FloatTensor(P)

if use_dist:
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank%torch.cuda.device_count())
    device = torch.device("cuda", local_rank)
else:
    local_rank = 0
device = 'cuda:0'
# model = VSR_CAS(channel0=opt.ChDim, factor=opt.upscale_factor, P=P ,patch_size =opt.patch_size).cuda()
model = DGSMPIR(opt.ChDim, opt.upscale_factor).to(device)

if use_dist:
    model = torch.nn.parallel.DistributedDataParallel(model,
                                                        find_unused_parameters=True,
                                                      device_ids=[local_rank],
                                                      output_device=local_rank)
optimizer = optim.Adam(model.parameters(), lr=opt.lr)
scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.5)

if opt.nEpochs != 0:
    load_dict = torch.load(opt.save_folder+"_epoch_{}.pth".format(opt.nEpochs))
    opt.lr = load_dict['lr']
    epoch = load_dict['epoch']
    model.load_state_dict(load_dict['param'])
    optimizer.load_state_dict(load_dict['adam'])


criterion = nn.L1Loss()

def add_gaussian_noise(image_tensor, sigma):
    noise = torch.randn_like(image_tensor) * sigma/255.0
    noisy_image_tensor = image_tensor + noise.cuda()
    noisy_image_tensor = torch.clamp(noisy_image_tensor, 0, 1)  # 将像素值裁剪到 [0, 1] 范围内

    return noisy_image_tensor

current_time = datetime.now().strftime('%b%d_%H-%M-%S')
CURRENT_DATETIME_HOSTNAME = '/' + current_time + '_' + socket.gethostname()
tb_logger = SummaryWriter(log_dir='./tb_logger/' + 'unfolding2' + CURRENT_DATETIME_HOSTNAME)
current_step = 0


def mkdir(path):
    folder = os.path.exists(path)

    if not folder:
        os.makedirs(path)
        print("---  new folder...  ---")
        print("---  " + path + "  ---")
    else:
        print("---  There is " + path + " !  ---")


def train(epoch, optimizer, scheduler):
    return epoch 

def test():
    avg_psnr = 0
    avg_time = 0
    mkdir(opt.outputpath)
    model.eval()
    with torch.no_grad():
        for batch in testing_data_loader:
            Y, Z, X = batch[0].cuda(), batch[1].cuda(), batch[2].cuda()
            Y = Variable(Y).float()
            Z = Variable(Z).float()
            X = Variable(X).float()
            Y = add_gaussian_noise(Y, 10)
            Z = add_gaussian_noise(Z, 10)
            torch.cuda.synchronize()
            start_time = time.time()

            HX = model(Y, Z, X)
            # HX = HX[0]
            # HX=F.interpolate(Y, scale_factor=8, mode='bicubic')
            torch.cuda.synchronize()
            end_time = time.time()

            X = torch.squeeze(X).permute(1, 2, 0).cpu().numpy()
            HX = torch.squeeze(HX).permute(1, 2, 0).cpu().numpy()

            psnr = MPSNR(HX,X)
            im_name = batch[3][0]
            print(im_name)
            print(end_time - start_time)
            avg_time += end_time - start_time
            (path, filename) = os.path.split(im_name)
            io.savemat(opt.outputpath + filename, {'HX': HX})
            avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))
    print("===> Avg. time: {:.4f} s".format(avg_time / len(testing_data_loader)))
    return avg_psnr / len(testing_data_loader)


def checkpoint(epoch):
    mkdir(opt.save_folder)
    model_out_path = opt.save_folder+"_epoch_{}.pth".format(epoch)
    if epoch % 1 == 0:
        save_dict = dict(
            lr = optimizer.state_dict()['param_groups'][0]['lr'],
            param = model.state_dict(),
            adam = optimizer.state_dict(),
            epoch = epoch
        )
        torch.save(save_dict, model_out_path)

        print("Checkpoint saved to {}".format(model_out_path))

if opt.mode == 1:
    for epoch in range(opt.nEpochs + 1, 201):
        avg_loss = train(epoch, optimizer, scheduler)
        checkpoint(epoch)
        torch.cuda.empty_cache()
        scheduler.step()

else:
    for i in range(opt.nEpochs, opt.nEpochs+1):
        load_dict = torch.load(opt.save_folder + "_epoch_{}.pth".format(i))
        model.load_state_dict(load_dict['param'])
        test()
