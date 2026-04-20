import SimpleITK as sitk
import os
import torch
import numpy as np
import torch.nn as nn
from time import time
from torchvision.transforms import transforms
from torch.utils.data import Dataset
from torch import optim
from torch.utils.data import DataLoader

def make_dataset3d(mra_dir,seg_dir):
    imgs=[]
    masks=[]
    for mra_file in os.listdir(mra_dir):
        img=os.path.join(mra_dir,mra_file)
        imgs.append(img)
        mask = os.path.join(seg_dir, mra_file.replace('data-', 'seg-'))
        masks.append(mask)
    return imgs,masks

class VesselDataSet(Dataset):
    '''定义数据集'''
    def __init__(self, mra_dir,seg_dir, transform=None, target_transform=None):
        # 初始化文件路径或文件名列表。初始化该类的一些基本参数
        imgs, masks = make_dataset3d(mra_dir,seg_dir)
        self.imgs = imgs
        self.masks = masks
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        '''
        ＃1。从文件中读取一个数据（例如，使用numpy.fromfile，PIL.Image.open）。
         ＃2。预处理数据（例如torchvision.Transform）。
         ＃3。返回数据对（例如图像和标签）。
        #这里需要注意的是，第一步：read one data，是一个data
        '''
        x_path, y_path = self.imgs[index],self.masks[index]
        img_x = sitk.GetArrayFromImage(sitk.ReadImage(x_path)).astype(np.float32)
        img_y = sitk.GetArrayFromImage(sitk.ReadImage(y_path)).astype(np.float32)
        #img_x = sitk.GetArrayFromImage(sitk.ReadImage(x_path)).transpose((1, 2, 0))
        #img_y = sitk.GetArrayFromImage(sitk.ReadImage(y_path)).transpose((1, 2, 0))

        if self.transform is not None:
            img_x = self.transform(img_x)
            if img_x.shape[0]!=1:
                img_x=torch.unsqueeze(img_x, dim=0)

        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
            if img_y.shape[0] != 1:
                img_y = torch.unsqueeze(img_y, dim=0)
        return img_x, img_y, x_path

    def __len__(self):
        # 您应该将0更改为数据集的总大小。
        return len(self.imgs)

def make_dataset3d_IXI(mra_dir,seg_dir):
    imgs=[]
    masks=[]
    for mra_file in os.listdir(mra_dir):
        base_name = mra_file.split('.')[0]
        parts = base_name.split('_', 1)  # 只分割第一个下划
        project_name = parts[0]  # "IXI035-IOP-0873-MRA"
        patch_info = parts[1]  # "0_128_512"

        img=os.path.join(mra_dir,mra_file)
        imgs.append(img)
        mask = os.path.join(seg_dir,project_name+"_GT_"+ patch_info+".nii.gz")
        masks.append(mask)
    return imgs,masks

class IXIDataSet(Dataset):
    '''定义数据集'''
    def __init__(self, mra_dir,seg_dir, transform=None, target_transform=None):
        # 初始化文件路径或文件名列表。初始化该类的一些基本参数
        imgs, masks = make_dataset3d_IXI(mra_dir,seg_dir)
        self.imgs = imgs
        self.masks = masks
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        '''
        ＃1。从文件中读取一个数据（例如，使用numpy.fromfile，PIL.Image.open）。
         ＃2。预处理数据（例如torchvision.Transform）。
         ＃3。返回数据对（例如图像和标签）。
        #这里需要注意的是，第一步：read one data，是一个data
        '''
        x_path, y_path = self.imgs[index],self.masks[index]
        img_x = sitk.GetArrayFromImage(sitk.ReadImage(x_path)).astype(np.float32)
        img_y = sitk.GetArrayFromImage(sitk.ReadImage(y_path)).astype(np.float32)
        #img_x = sitk.GetArrayFromImage(sitk.ReadImage(x_path)).transpose((1, 2, 0))
        #img_y = sitk.GetArrayFromImage(sitk.ReadImage(y_path)).transpose((1, 2, 0))

        if self.transform is not None:
            img_x = self.transform(img_x)
            if img_x.shape[0]!=1:
                img_x=torch.unsqueeze(img_x, dim=0)

        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
            if img_y.shape[0] != 1:
                img_y = torch.unsqueeze(img_y, dim=0)
        return img_x, img_y, x_path

    def __len__(self):
        # 您应该将0更改为数据集的总大小。
        return len(self.imgs)

def make_dataset3d_CAS2023(mra_dir,seg_dir):
    imgs=[]
    masks=[]
    for mra_file in os.listdir(mra_dir):
        img=os.path.join(mra_dir,mra_file)
        imgs.append(img)
        mask = os.path.join(seg_dir, mra_file)
        masks.append(mask)
    return imgs,masks

class VesselDataSet_CAS2023(Dataset):
    '''定义数据集'''
    def __init__(self, mra_dir,seg_dir, transform=None, target_transform=None):
        # 初始化文件路径或文件名列表。初始化该类的一些基本参数
        imgs, masks = make_dataset3d_CAS2023(mra_dir,seg_dir)
        self.imgs = imgs
        self.masks = masks
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        '''
        ＃1。从文件中读取一个数据（例如，使用numpy.fromfile，PIL.Image.open）。
         ＃2。预处理数据（例如torchvision.Transform）。
         ＃3。返回数据对（例如图像和标签）。
        #这里需要注意的是，第一步：read one data，是一个data
        '''
        x_path, y_path = self.imgs[index],self.masks[index]
        img_x = sitk.GetArrayFromImage(sitk.ReadImage(x_path)).astype(np.float32)
        img_y = sitk.GetArrayFromImage(sitk.ReadImage(y_path)).astype(np.float32)
        #img_x = sitk.GetArrayFromImage(sitk.ReadImage(x_path)).transpose((1, 2, 0))
        #img_y = sitk.GetArrayFromImage(sitk.ReadImage(y_path)).transpose((1, 2, 0))

        if self.transform is not None:
            img_x = self.transform(img_x)
            if img_x.shape[0]!=1:
                img_x=torch.unsqueeze(img_x, dim=0)

        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
            if img_y.shape[0] != 1:
                img_y = torch.unsqueeze(img_y, dim=0)
        return img_x, img_y, x_path

    def __len__(self):
        # 您应该将0更改为数据集的总大小。
        return len(self.imgs)

def make_dataset3d_CTA(mra_dir,seg_dir):
    imgs=[]
    masks=[]
    for mra_file in os.listdir(mra_dir):
        img=os.path.join(mra_dir,mra_file)
        imgs.append(img)
        mask = os.path.join(seg_dir, mra_file.replace('data-', 'label-'))
        masks.append(mask)
    return imgs,masks

class CTAVesselDataSet(Dataset):
    '''定义数据集'''
    def __init__(self, mra_dir,seg_dir, transform=None, target_transform=None):
        # 初始化文件路径或文件名列表。初始化该类的一些基本参数
        imgs, masks = make_dataset3d_CTA(mra_dir,seg_dir)
        self.imgs = imgs
        self.masks = masks
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        '''
        ＃1。从文件中读取一个数据（例如，使用numpy.fromfile，PIL.Image.open）。
         ＃2。预处理数据（例如torchvision.Transform）。
         ＃3。返回数据对（例如图像和标签）。
        #这里需要注意的是，第一步：read one data，是一个data
        '''
        x_path, y_path = self.imgs[index],self.masks[index]
        img_x = sitk.GetArrayFromImage(sitk.ReadImage(x_path)).astype(np.float32)
        img_y = sitk.GetArrayFromImage(sitk.ReadImage(y_path)).astype(np.float32)
        #img_x = sitk.GetArrayFromImage(sitk.ReadImage(x_path)).transpose((1, 2, 0))
        #img_y = sitk.GetArrayFromImage(sitk.ReadImage(y_path)).transpose((1, 2, 0))

        if self.transform is not None:
            img_x = self.transform(img_x)
            if img_x.shape[0]!=1:
                img_x=torch.unsqueeze(img_x, dim=0)

        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
            if img_y.shape[0] != 1:
                img_y = torch.unsqueeze(img_y, dim=0)
        return img_x, img_y, x_path

    def __len__(self):
        # 您应该将0更改为数据集的总大小。
        return len(self.imgs)




def make_dataset3d_CTAall(mra_dir,seg_dir):
    imgs=[]
    masks=[]
    for mra_file in os.listdir(mra_dir):
        img=os.path.join(mra_dir,mra_file)
        imgs.append(img)
        mask = os.path.join(seg_dir, mra_file.replace('data-', 'label-'))
        masks.append(mask)
    return imgs,masks

class VesselDataSet_CTAall(Dataset):
    '''定义数据集'''
    def __init__(self, mra_dir,seg_dir, transform=None, target_transform=None):
        # 初始化文件路径或文件名列表。初始化该类的一些基本参数
        imgs, masks = make_dataset3d_CTAall(mra_dir,seg_dir)
        self.imgs = imgs
        self.masks = masks
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        '''
        ＃1。从文件中读取一个数据（例如，使用numpy.fromfile，PIL.Image.open）。
         ＃2。预处理数据（例如torchvision.Transform）。
         ＃3。返回数据对（例如图像和标签）。
        #这里需要注意的是，第一步：read one data，是一个data
        '''
        x_path, y_path = self.imgs[index],self.masks[index]
        img_x = sitk.GetArrayFromImage(sitk.ReadImage(x_path)).astype(np.float32)
        img_y = sitk.GetArrayFromImage(sitk.ReadImage(y_path)).astype(np.float32)
        #img_x = sitk.GetArrayFromImage(sitk.ReadImage(x_path)).transpose((1, 2, 0))
        #img_y = sitk.GetArrayFromImage(sitk.ReadImage(y_path)).transpose((1, 2, 0))

        if self.transform is not None:
            img_x = self.transform(img_x)
            if img_x.shape[0]!=1:
                img_x=torch.unsqueeze(img_x, dim=0)

        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
            if img_y.shape[0] != 1:
                img_y = torch.unsqueeze(img_y, dim=0)
        return img_x, img_y, x_path

    def __len__(self):
        # 您应该将0更改为数据集的总大小。
        return len(self.imgs)














if __name__=="__main__":
    x_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])
    y_transforms = transforms.ToTensor()
    batch_size=1
    num_epoch = 1

    dataset = VesselDataSet(r"E:\boold_c\tum_patch\data",
                        r"E:\boold_c\tum_patch\seg", transform=x_transforms,
                        target_transform=y_transforms)
    # dataset = VesselDataSet("C:\\Users\\englishrenj\\Desktop\\net\\patch\\2d\\128\\data\\",
    #                         "C:\\Users\\englishrenj\\Desktop\\net\\patch\\2d\\128\\seg\\", transform=x_transforms,
    #                         target_transform=y_transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)


    for epoch in range(num_epoch):
        print('Epoch {}/{}'.format(epoch, num_epoch - 1))
        print('-------------------------------------')

        #注意2维切片少一个维度
        for x, y,path in dataloader:
            print('data-------------------------------------')
            print('size:{0},{1},{2},{3},{4}'.format(x.shape[0],x.shape[1],x.shape[2],x.shape[3],x.shape[4]))
            print('seg-------------------------------------')
            print('size:{0},{1},{2},{3},{4}'.format(y.shape[0], y.shape[1], y.shape[2], y.shape[3], y.shape[4]))
            print(path[0].replace('data', 'pre'))
            print('__________________________________________')

