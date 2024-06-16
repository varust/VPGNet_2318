from os.path import exists, join, basename
from os import makedirs, remove
from six.moves import urllib
import tarfile

from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize

from dataset import DatasetFromFolder, DatasetFromFolders
from dataset import DatasetFromFolder2, DatasetFromFolder2s
from torch.utils.data import DataLoader


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)

def input_transform():
    return Compose([
        ToTensor(),
    ])


def get_patch_training_set(upscale_factor, patch_size, len=20000, srgb=False):
    root_dir = "/data1/HSISR/CAVE/"
    train_dir1 = join(root_dir, "train/HSI")
    train_dir2 = join(root_dir, "train/RGB")
    train_dir3 = join(root_dir, "train/LR")
    if srgb:
        train_dir4 = join(root_dir, "train/stable_RGB")
        return DatasetFromFolders(train_dir1, train_dir2, train_dir3, train_dir4, upscale_factor, patch_size, lens=len,
                                 input_transform=input_transform())
    else:
        return DatasetFromFolder(train_dir1,train_dir2,train_dir3,upscale_factor, patch_size, lens=len, input_transform=input_transform())


def get_test_set(srgb=False):
    root_dir = "/data1/HSISR/CAVE/"
    test_dir1 = join(root_dir, "test/HSI")
    test_dir2 = join(root_dir, "test/RGB")
    test_dir3 = join(root_dir, "test/LR")
    if srgb:
        test_dir4 = join(root_dir, "test/stable_RGB")
        return DatasetFromFolder2s(test_dir1, test_dir2, test_dir3, test_dir4, input_transform=input_transform())
    else:
        return DatasetFromFolder2(test_dir1,test_dir2,test_dir3, input_transform=input_transform())


