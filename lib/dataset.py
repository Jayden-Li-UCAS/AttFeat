import numpy as np
from PIL import Image
import os
import torch
from torch.utils.data import Dataset
from lib.utils import *
from torchvision import transforms

#######dataset,loss,utils
##根据rot_mode进行图像的不同角度旋转
##这样通过旋转图像,可以增加训练样本的多样性,提高模型对旋转的鲁棒性,是计算机视觉任务中的一个有效数据增广技巧。
def rot(img, rot_mode):
    ##当rot_mode=0时,先进行矩阵转置,再水平翻转,实现图像逆时针旋转90度。
    if rot_mode == 0:
        img = img.transpose(1, 2)
        img = img.flip(1)
    ##当rot_mode=1时,先水平翻转,再垂直翻转,实现图像旋转180度。
    elif rot_mode == 1:
        img = img.flip(1)
        img = img.flip(2)
    ##当rot_mode=2时,先水平翻转,再矩阵转置,实现图像顺时针旋转90度。
    elif rot_mode == 2:
        img = img.flip(1)
        img = img.transpose(1, 2)
    return img

##根据flip_mode进行图像的水平或者垂直翻转
def flip(img, flip_mode):
    ##flip_mode为0表示水平翻转
    if flip_mode == 0:
        img = img.flip(1)
    ##flip_mode为1表示垂直翻转
    elif flip_mode == 1:
        img = img.flip(2)
    return img



class MMDataset(Dataset):
    def __init__(self,path,type,transform=None,crop_size=480):   ##480
        # fh = open(txt, 'r')
        path0=os.path.join(path,type,'train',type.split('_')[0])
        path1=os.path.join(path,type,'train',type.split('_')[1])
        pathDir = os.listdir(path0)
        imgs0 = []
        imgs1 = []
        for i in range(len(pathDir)):
            imgs0.append(path0+'/'+pathDir[i])
            imgs1.append(path1+'/'+pathDir[i])
        self.imgs0 = imgs0
        self.imgs1 = imgs1
        self.transform = transform
        self.crop_size = crop_size
        ##在输入图像上加入一定模糊和噪声,一方面有利于增强模型的泛化能力,防止模型过度依赖细节特征;另一方面也提供一定数据增广的作用
        self.suffix_transform = transforms.Compose([
                                                    transforms.GaussianBlur(kernel_size=[3],sigma=[0.01,1]),
                                                    transforms.Lambda(RandomNoise)])
        # self.loader = loader

    def __getitem__(self, index):
        img0 = Image.open(self.imgs0[index])
        if img0.mode != 'RGB':
            img0 = img0.convert('RGB')
        img1 = Image.open(self.imgs1[index])
        if img1.mode != 'RGB':
            img1 = img1.convert('RGB')
       
        img0 = self.transform(img0)   ##3,343,586      144
        img1 = self.transform(img1)   ##3,343,586
        ####在输入图像上加入一定模糊和噪声,一方面有利于增强模型的泛化能力,防止模型过度依赖细节特征;另一方面也提供一定数据增广的作用
        img0 = self.suffix_transform(img0)
        img1 = self.suffix_transform(img1)
        flip_mode = np.random.randint(0, 3)  ##0,水平翻转  1，垂直翻转  2，不翻转
        rot_mode = np.random.randint(0, 4)   ##0,90度   1，180度   2，270度   3，360度即0度
        ####根据flip_mode进行图像的水平或者垂直翻转
        img0 = flip(img0,flip_mode)  ##3,586,343
        img1 = flip(img1,flip_mode)  ##3,586,343
        ##根据rot_mode进行图像的不同角度旋转，90，180，270，360
        img0 = rot(img0,rot_mode)
        img1 = rot(img1,rot_mode)
        ##对图像进行归一化:减去均值,除以标准差,这可以消除图像的平均亮度和对比度影响。
        img0 = (img0-img0.mean(dim=[-1,-2],keepdim=True))/(img0.std(dim=[-1,-2],keepdim=True)+1e-5)
        img1 = (img1-img1.mean(dim=[-1,-2],keepdim=True))/(img1.std(dim=[-1,-2],keepdim=True)+1e-5)
        img0, img1, aflow = Random_proj(img0,img1,self.crop_size)
        return {
            'img1': torch.FloatTensor(img0),
            'img2': torch.FloatTensor(img1),
            'aflow':aflow
        }
         

    def __len__(self):
        return len(self.imgs0)