"""
温医血管数据预处理
1) 查看数据像素值的分布直方图
2) 将数据用 zscore标准化，也有用 minmax归一化的做法，不确定那种做法好。 看到有说 zscore标准化可以防止数据规范化时被压缩

3) 统计数据的space
4) 将数据重采样

5) 将train 数据分patch，剔除掉不包含血管数据的patch。   正负样本不均衡，这样做应该有点用
6) 
7)
"""

import pathlib
#import ants
from matplotlib import pyplot as plt
import numpy as np
import os
import SimpleITK as sitk
import pandas as pd
import nibabel as nib
from scipy import ndimage
from regex import F
import glob
import shutil
import random

def clean_Data(path):
    save_name = ["label.nii.gz", "T1.nii.gz", "T1_enhance.nii.gz", "T2.nii.gz"]
    filelists = os.listdir(path)
    for filelist in filelists:
        if filelist not in save_name:
            # print("remove:", path + '/' + filelist)
            os.remove(path + '/' + filelist)



'''
def registration_nii(target_path, save_path):
    fix_name = "T1_enhance.nii.gz"
    fix_img = ants.image_read(os.path.join(target_path, fix_name))
    move_list = ["T1.nii.gz", "T2.nii.gz"]
  
    for move_name in move_list:
        move_img = ants.image_read(os.path.join(target_path, move_name))
        outs = ants.registration(fix_img, move_img, type_of_transforme = 'SyN')  
        reg_img = outs['warpedmovout']  
        out_name = move_name.replace('.nii.gz', '.nii.gz')
        ants.image_write(reg_img,os.path.join(save_path, out_name))
 
    out_name = fix_name.replace('.nii.gz', '.nii.gz')
    ants.image_write(fix_img,os.path.join(save_path, out_name))


    # resamplemethod = sitk.sitkNearestNeighbor
    # label_name = 'label.nii'
    # ori_img = sitk.ReadImage(os.path.join(target_path, label_name))

    # target_img = sitk.ReadImage(os.path.join(target_path, fix_name))
    # target_Size = target_img.GetSize()  # 目标图像大小  [x,y,z]
    # target_Spacing = target_img.GetSpacing()  # 目标的体素块尺寸    [x,y,z]
    # target_origin = target_img.GetOrigin()  # 目标的起点 [x,y,z]
    # target_direction = target_img.GetDirection()  # 目标的方向 [冠,矢,横]=[z,y,x]
    #     # itk的方法进行resample
    # resampler = sitk.ResampleImageFilter()

    # #resampler.SetReferenceImage(ori_img)  # 需要重新采样的目标图像
    #     # 设置目标图像的信息
    # resampler.SetSize(target_Size)  # 目标图像大小
    # resampler.SetOutputOrigin(target_origin)
    # resampler.SetOutputDirection(target_direction)
    # resampler.SetOutputSpacing(target_Spacing)
    #     # 根据需要重采样图像的情况设置不同的dype
    # if resamplemethod == sitk.sitkNearestNeighbor:
    #     resampler.SetOutputPixelType(sitk.sitkUInt8)  # 近邻插值用于mask的，保存uint8
    # else:
    #     resampler.SetOutputPixelType(sitk.sitkFloat32)  # 线性插值用于PET/CT/MRI之类的，保存float32

    # resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    # resampler.SetInterpolator(resamplemethod)
    # itk_img_resampled = resampler.Execute(ori_img)  # 得到重新采样后的图像
    # out_name = label_name.replace('.nii', '.nii.gz')
    # path_name = os.path.join(save_path, out_name)
    # sitk.WriteImage(itk_img_resampled, path_name)
'''





def resampleVolume_nii(target_path, save_path):
    outspacing = [0.7,0.7,0.6]
    outsize = [0, 0, 0]
    nii_list = os.listdir(target_path)
    for nii_name in nii_list:
        resapmler_img = sitk.ReadImage(os.path.join(target_path, nii_name))
        inputsize = resapmler_img.GetSize()
        inputspacing = resapmler_img.GetSpacing()
        transform = sitk.Transform()
        transform.SetIdentity()
        outsize[0] = int(inputsize[0] * inputspacing[0] / outspacing[0] + 0.5)
        outsize[1] = int(inputsize[1] * inputspacing[1] / outspacing[1] + 0.5)
        outsize[2] = int(inputsize[2] * inputspacing[2] / outspacing[2] + 0.5)

        resampler = sitk.ResampleImageFilter()
        resampler.SetTransform(transform)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetOutputOrigin(resapmler_img.GetOrigin())
        resampler.SetOutputSpacing(outspacing)
        resampler.SetOutputDirection(resapmler_img.GetDirection())
        resampler.SetSize(outsize)
        save_img = resampler.Execute(resapmler_img)
        path_name = os.path.join(save_path, nii_name)
        sitk.WriteImage(save_img, path_name)




def get_info(path):
    folds = os.listdir(path)
    t1_size = []
    t1ce_size = []
    t2_size = []
    t1_space = []
    t1ce_space = []
    t2_space = []

    for fold in folds:
        img_name = os.listdir(os.path.join(path, fold))
        for img in img_name:
            if img == 'T1.nii.gz' :
                target= sitk.ReadImage(os.path.join(path, fold, img))
                t1_size.append(target.GetSize())
                t1_space.append(target.GetSpacing())
                print("Get:" , img)
            elif img == 'T1_enhance.nii.gz':
                target= sitk.ReadImage(os.path.join(path, fold, img))
                t1ce_size.append(target.GetSize())
                t1ce_space.append(target.GetSpacing())
                print("Get:" , img)
            elif img == 'T2.nii.gz':
                target= sitk.ReadImage(os.path.join(path, fold, img))
                t2_size.append(target.GetSize())
                t2_space.append(target.GetSpacing())
                print("Get:" , img)
    dataframe = pd.DataFrame({'t1_size': t1_size,'t1ce_size': t1ce_size, 't2_size':t2_size, 't1_space':t1_space, 't1ce_space':t1ce_space, 't2_space': t2_space})
    dataframe.to_csv("niidata_info.csv", index=False)


'''
def n4_correction(target_path, save_path):
    nii_list = ['T1.nii.gz','T1_enhance.nii.gz', 'T2.nii.gz']
    #label_name = 'label.nii.gz'
    #label = ants.image_read(os.path.join(target_path, label_name))
    #ants.image_write(label, os.path.join(save_path, label_name))
    for nii_name in nii_list:
        image = ants.image_read(os.path.join(target_path, nii_name))
        image_n4 = ants.n4_bias_field_correction(image) 
        ants.image_write(image_n4, os.path.join(save_path, nii_name))
'''


def fill_hols(path):
    label_name = 'label.nii.gz'
    folds = os.listdir(path)
    for fold in folds:
        img_nii = sitk.ReadImage(os.path.join(path, fold, label_name), outputPixelType=sitk.sitkUInt16)
        img_fill = sitk.BinaryFillhole(img_nii)
        save_name = label_name.replace('.nii.gz', '_fill.nii.gz')
        img_savedir = os.path.join(path, fold, save_name)
        sitk.WriteImage(img_fill, img_savedir)

        
## -----------------------------------------------------------------------
## --- 统计图像的 size, origin、spacing 和 direction 信息 --- ##
def get_image_info(data_path):
    data_path = pathlib.Path(data_path).resolve()
    assert data_path.exists()

    img_dir = os.path.join(data_path, 'image')
    label_dir = os.path.join(data_path, 'label')

    image_list = sorted([os.path.join(img_dir, x) for x in os.listdir(img_dir) if x.endswith('.nii.gz')])
    label_list = sorted([os.path.join(label_dir, x) for x in os.listdir(label_dir) if x.endswith('.nii.gz')])

    print("----- start get image info -----")
    image_info = []
    for image_path, label_path in zip(image_list, label_list):
        image = sitk.ReadImage(image_path)
        label = sitk.ReadImage(label_path)

        origin_image = image.GetOrigin()
        spacing_image = image.GetSpacing()
        direction_image = image.GetDirection()
        size_image = image.GetSize()

        origin_label = label.GetOrigin()
        spacing_label = label.GetSpacing()
        direction_label = label.GetDirection()
        size_label = label.GetSize()

        image_info.append({
            'Image Name': os.path.basename(image_path),
            'Image Origin': origin_image,
            'Image Spacing': spacing_image,
            'Image Direction': direction_image,
            'Image Size': size_image,
            'Label Origin': origin_label,
            'Label Spacing': spacing_label,
            'Label Direction': direction_label,
            'Label Size': size_label
        })

    df = pd.DataFrame(image_info)

    # Determine the output path and filename based on the input data path
    data_folder_name = os.path.basename(os.path.normpath(data_path))
    output_excel_path = os.path.join(data_path, f'{data_folder_name}_info.xlsx')
    df.to_excel(output_excel_path, index=False)
    print(f"-----get {data_folder_name}_info successful !!! -----")



## ---  数据的像素值直方图绘制 --- ## 
def get_histograms(data_path):
    data_path = pathlib.Path(data_path).resolve()
    assert data_path.exists()

    img_dir = data_path / 'image'
    image_filenames = sorted([x for x in img_dir.iterdir() ])

    histograms_output_path = os.path.join(data_path, 'histograms')
    os.makedirs(histograms_output_path, exist_ok=True)

    ## --- 像素值统计 --- ##
    pixel_values_list = []
    for image_path in image_filenames:
        image = sitk.GetArrayFromImage(sitk.ReadImage(str(image_path)))
        non_zero_pixel_values = image.flatten()[image.flatten() != 0]  # 排除像素值为0的部分
        pixel_values_list.append(non_zero_pixel_values)

    ## --- 绘制直方图 --- ##
    print("----- start plot histograms -----")
    for i, (image_path, pixel_values) in enumerate(zip(image_filenames, pixel_values_list)):
        plt.figure(figsize=(8, 6))
        min_val = np.min(pixel_values)
        max_val = np.max(pixel_values)
        plt.hist(pixel_values, bins=100, range=(min_val, max_val), alpha=0.7)
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        title = f"{image_path.stem}_Pixel Value Histogram"
        plt.title(title)
        plt.text(0.75, 0.85, f"max pixel = {max_val}", transform=plt.gca().transAxes)

        # 计算均值和方差
        mean_val = np.mean(pixel_values)
        std_val = np.std(pixel_values)
        plt.text(0.75, 0.75, f"mean = {mean_val:.2f}", transform=plt.gca().transAxes)
        plt.text(0.75, 0.65, f"std = {std_val:.2f}", transform=plt.gca().transAxes)

        output_filename = f"{image_path.stem}__histogram.png"
        output_path = os.path.join(histograms_output_path, output_filename)
        plt.savefig(output_path)
        plt.close()
    print("----- finish plot histograms -----")



## ----- 0背景去除 ----- ##
def crop_images_and_labels(image_list, label_list, output_path):
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'labels'), exist_ok=True)

    for image_path, label_path in zip(image_list, label_list):
        image = sitk.ReadImage(str(image_path))     ## --- W,H,D --- ##
        label = sitk.ReadImage(str(label_path))
        #print(image.GetSize())
        ##--- 读取方向，原点，空间等信息 ---##
        image_Direction = image.GetDirection()
        image_Origin = image.GetOrigin()
        image_Spacing = image.GetSpacing()

        label_Direction = label.GetDirection()
        label_Origin = label.GetOrigin()
        label_Spacing = label.GetSpacing()

        image_array = sitk.GetArrayFromImage(image)     ## --- D,H,W ---##
        label_array = sitk.GetArrayFromImage(label)

        # Find the min and max coordinates of the non-zero region in 3D space for this image
        z_indexes, y_indexes, x_indexes = np.nonzero(image_array != 0)
        zmin, ymin, xmin = [max(0, int(np.min(arr) - 1)) for arr in (z_indexes, y_indexes, x_indexes)]
        zmax, ymax, xmax = [int(np.max(arr) + 1) for arr in (z_indexes, y_indexes, x_indexes)]

        cropped_image_array = image_array[zmin:zmax, ymin:ymax, xmin:xmax]
        cropped_label_array = label_array[zmin:zmax, ymin:ymax, xmin:xmax]

        # Update the origin and spacing of the cropped image to match the original image
        cropped_image = sitk.GetImageFromArray(cropped_image_array)
        cropped_label = sitk.GetImageFromArray(cropped_label_array)

        cropped_image.SetDirection(image_Direction)
        cropped_image.SetOrigin(image_Origin)
        cropped_image.SetSpacing(image_Spacing)

        cropped_label.SetDirection(label_Direction)
        cropped_label.SetOrigin(label_Origin)
        cropped_label.SetSpacing(label_Spacing)


        # Save the cropped images and labels
        output_image_path = os.path.join(output_path, 'images', image_path.name)
        output_label_path = os.path.join(output_path, 'labels', label_path.name)

        sitk.WriteImage(cropped_image, output_image_path)
        sitk.WriteImage(cropped_label, output_label_path)

def remove_zero_background(raw_data_path):
    raw_data_path = pathlib.Path(raw_data_path).resolve()
    assert raw_data_path.exists()

    img_dir = raw_data_path / 'images'
    label_dir = raw_data_path / 'labels'

    image_filenames = sorted([x for x in img_dir.iterdir() ])
    label_filenames = sorted([x for x in label_dir.iterdir() ])
    print("----- strat remove_zero_background -----")
    
    # Output path for the cropped images and labels
    output_path = '/media/ssd1/ly/Vessel/Datasets/WenYi/preprocess_data/01_zero_background_removed'

    # Crop images and labels based on the non-zero region for each image
    crop_images_and_labels(image_filenames, label_filenames, output_path)

    print("----- Zero background removed and data saved successfully -----")
    return output_path


## ---  标准化/归一化 --- ##
def zscore_normalize(image_array):
    non_zero_pixels = image_array[image_array != 0]
    mean = np.mean(non_zero_pixels)
    std = np.std(non_zero_pixels)
    normalized_image_array = (image_array - mean) / std
    return normalized_image_array

def minmax_normalize(image_array):
    non_zero_pixels = image_array[image_array != 0]
    min_val = np.min(non_zero_pixels)
    max_val = np.max(non_zero_pixels)
    normalized_image_array = (image_array - min_val) / (max_val - min_val)
    return normalized_image_array

def normalize(input_path, normalization):
    input_path = pathlib.Path(input_path).resolve()
    assert input_path.exists()

    output_path = os.path.join("/home/segmamba", f"CAS2023_normalized_{normalization}")
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, 'data'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'mask'), exist_ok=True)

    img_dir = input_path / 'data'
    label_dir = input_path / 'mask'

    image_filenames = sorted([x for x in img_dir.iterdir() ])
    label_filenames = sorted([x for x in label_dir.iterdir() ])

    for image_path, label_path in zip(image_filenames, label_filenames):
        image = sitk.ReadImage(image_path)
        label = sitk.ReadImage(label_path)

        image_array = sitk.GetArrayFromImage(image)

        if normalization == "zscore":
            normalized_image_array = zscore_normalize(image_array)
        elif normalization == "minmax":
            normalized_image_array = minmax_normalize(image_array)
        else:
            raise ValueError("Invalid normalization method. Please choose 'zscore' or 'minmax'.")

        normalized_image = sitk.GetImageFromArray(normalized_image_array)
        normalized_image.SetSpacing(image.GetSpacing())
        normalized_image.SetOrigin(image.GetOrigin())
        normalized_image.SetDirection(image.GetDirection())

        output_image_path = os.path.join(output_path, 'data', image_path.name)
        output_label_path = os.path.join(output_path, 'mask', label_path.name)

        sitk.WriteImage(normalized_image, output_image_path)
        sitk.WriteImage(label, output_label_path)
    return output_path



## --- 重采样 --- ##




## --- 滑动分patch --- ##
def pad_to_fit_shape(data, target_shape, pad_way='constant'):
    # Calculate padding in each dimension
    padding = [(0, max(0, target_shape[i] - data.shape[i])) for i in range(len(target_shape))]
    # Pad the data according to the specified padding mode
    if pad_way == 'constant':
        padded_data = np.pad(data, padding, mode='constant')        ## --- 默认0值填充 --- ##
    elif pad_way == 'edge':
        padded_data = np.pad(data, padding, mode='edge')
    elif pad_way == 'reflect':
        padded_data = np.pad(data, padding, mode='reflect')
    elif pad_way == 'symmetric':
        padded_data = np.pad(data, padding, mode='symmetric')
    else:
        raise ValueError("Invalid pad_way. Please choose from 'constant', 'edge', 'reflect', or 'symmetric'.")

    return padded_data


def split_patch(input_path, HW_stride, D_stride, patch_size=(128, 128, 128), pad_way='constant'):
    input_path = pathlib.Path(input_path).resolve()
    assert input_path.exists()

    img_dir = input_path / 'data'
    label_dir = input_path / 'label'
    image_filenames = sorted([x for x in img_dir.iterdir()])
    label_filenames = sorted([x for x in label_dir.iterdir()])

    # output_path = os.path.join(input_path, f'split_patch_pad_with_{pad_way}', f"{HW_stride}_{D_stride}")
    #output_path = os.path.join('/home/segmamba/CAS2023_then_split/', f'split_patch_pad_with_{pad_way}', f"{HW_stride}_{D_stride}_size{patch_size[0]}")
    output_path = '/home/segmamba/CTAall_processed/train/'
    os.makedirs(output_path, exist_ok=True)
    images_output_path = os.path.join(output_path, 'data')
    labels_output_path = os.path.join(output_path, 'label')
    os.makedirs(images_output_path, exist_ok=True)
    os.makedirs(labels_output_path, exist_ok=True)

    print(f"----- strat split patch is {patch_size}, H_W stride is {HW_stride}, D stride is {D_stride} -----")

    for image_path, label_path in zip(image_filenames, label_filenames):
        print(f'Start split {image_path}, {label_path}')
        image = sitk.ReadImage(image_path)
        label = sitk.ReadImage(label_path)

        ##--- 读取方向，原点，空间等信息 ---##
        image_Direction = image.GetDirection()
        image_Origin = image.GetOrigin()
        image_Spacing = image.GetSpacing()

        label_Direction = label.GetDirection()
        label_Origin = label.GetOrigin()
        label_Spacing = label.GetSpacing()
        ##判断image和label的方向、原点、空间信息是否一致
        # if image.GetDirection() != label.GetDirection():
        #     raise ValueError("Image and label information (direction) must be the same. "
        #                      f"Error in data: {image_path}, {label_path}")
        # if image.GetOrigin() != label.GetOrigin():
        #     raise ValueError("Image and label information (origin) must be the same. "
        #                      f"Error in data: {image_path}, {label_path}")
        # if image.GetSpacing() != label.GetSpacing():
        #     raise ValueError("Image and label information (spacing) must be the same. "
        #                      f"Error in data: {image_path}, {label_path}")

        image_arry = sitk.GetArrayFromImage(image)
        label_arry = sitk.GetArrayFromImage(label)

        D, H, W = image_arry.shape
        print(image_arry.shape)
        D_l, H_l, W_l = label_arry.shape
        # 判断image和label的大小是否一致
        if (D, H, W) != (D_l, H_l, W_l):
            raise ValueError("Image and label dimensions must be the same."
                             f"Error in data: {image_path}, {label_path}")

        d_pad = D_stride - (D % D_stride) if D % D_stride != 0 else 0
        h_pad = HW_stride - (H % HW_stride) if H % HW_stride != 0 else 0
        w_pad = HW_stride - (W % HW_stride) if W % HW_stride != 0 else 0

        after_pad_image_arry = pad_to_fit_shape(image_arry, (D + d_pad, H + h_pad, W + w_pad), pad_way=pad_way)
        after_pad_label_arry = pad_to_fit_shape(label_arry, (D + d_pad, H + h_pad, W + w_pad), pad_way=pad_way)  

        # image_output_path = os.path.join(output_path, "data", image_path.stem.split('.')[0])
        # label_output_path = os.path.join(output_path, "label", label_path.stem.split('.')[0])
        image_output_path = os.path.join(output_path, "data")
        label_output_path = os.path.join(output_path, "label")
        os.makedirs(image_output_path, exist_ok=True)
        os.makedirs(label_output_path, exist_ok=True)

        for d_start in range(0, after_pad_image_arry.shape[0] - patch_size[0] + 1, D_stride):
            for h_start in range(0, after_pad_image_arry.shape[1] - patch_size[1] + 1, HW_stride):
                for w_start in range(0, after_pad_image_arry.shape[2] - patch_size[2] + 1, HW_stride):
                    image_patch_arry = after_pad_image_arry[d_start:d_start + patch_size[0],
                                        h_start:h_start + patch_size[1],
                                        w_start:w_start + patch_size[2]]
                    label_patch_arry = after_pad_label_arry[d_start:d_start + patch_size[0],
                                        h_start:h_start + patch_size[1],
                                        w_start:w_start + patch_size[2]]

                    # Save the patch as a new image
                    image_patch_name = f"{image_path.stem.split('.')[0]}_{d_start}_{h_start}_{w_start}.nii.gz"
                    label_patch_name = f"{label_path.stem.split('.')[0]}_{d_start}_{h_start}_{w_start}.nii.gz"

                    image_patch_path = os.path.join(image_output_path, image_patch_name)
                    label_patch_path = os.path.join(label_output_path, label_patch_name)

                    image_patch = sitk.GetImageFromArray(image_patch_arry)
                    label_patch = sitk.GetImageFromArray(label_patch_arry)

                    image_patch.SetDirection(image_Direction)
                    image_patch.SetOrigin(image_Origin)
                    image_patch.SetSpacing(image_Spacing)

                    label_patch.SetDirection(label_Direction)
                    label_patch.SetOrigin(label_Origin)
                    label_patch.SetSpacing(label_Spacing)

                    sitk.WriteImage(image_patch, image_patch_path)
                    sitk.WriteImage(label_patch, label_patch_path)
    print("----- finish split patch successfully -----")
    return output_path


## --- 根据label的像素筛选patch --- ##
def filter_patches(input_path, threshold):
    input_path = pathlib.Path(input_path).resolve()
    assert input_path.exists()

    img_dir = input_path / 'data'
    label_dir = input_path / 'mask'
    image_filenames = sorted([x for x in img_dir.iterdir()])
    label_filenames = sorted([x for x in label_dir.iterdir()])

    output_path = os.path.join('/home/segmamba/', f'CAS2023_filter_threshold_{threshold}')
    images_output_path = os.path.join(output_path, 'data')
    labels_output_path = os.path.join(output_path, 'mask')
    os.makedirs(images_output_path, exist_ok=True)
    os.makedirs(labels_output_path, exist_ok=True)

    for image_path, label_path in zip(image_filenames, label_filenames):
        images = sorted([x for x in image_path.iterdir()])
        labels = sorted([x for x in label_path.iterdir()])

        image_output_path = os.path.join(images_output_path, image_path.stem.split('.')[0])
        label_output_path = os.path.join(labels_output_path, label_path.stem.split('.')[0])
        os.makedirs(image_output_path, exist_ok=True)
        os.makedirs(label_output_path, exist_ok=True)

        num_valid_patches = 0
        for image, label in zip(images, labels):  
            image_patch = nib.load(image).get_fdata()
            label_patch = nib.load(label).get_fdata()

            num_valid_pixels = np.count_nonzero(label_patch)
            
            if num_valid_pixels >= threshold:
                num_valid_patches += 1
                # Save the selected patch to the output directory
                shutil.copy(image, os.path.join(image_output_path, image.name))
                shutil.copy(label, os.path.join(label_output_path, label.name))
                
        print(f"Subdirectory {image_path}: {num_valid_patches} patches selected and saved")




## --- 划分数据集 --- ##

def split_dataset(train_data_input_path, test_data_input_path):
    random.seed(42)     ## 设置随机数种子

    # 设置路径
    train_images_path = '/home/segmamba/CTA_data_preprocess/train_128_2000/image'
    train_labels_path = '/home/segmamba/CTA_data_preprocess/train_128_2000/label'
    test_images_path = '/home/segmamba/CTA_data_preprocess/val_128_2000/image'
    test_labels_path = '/home/segmamba/CTA_data_preprocess/val_128_2000/label'

    # 创建输出文件夹
    os.makedirs(train_images_path, exist_ok=True)
    os.makedirs(train_labels_path, exist_ok=True)
    os.makedirs(test_images_path, exist_ok=True)
    os.makedirs(test_labels_path, exist_ok=True)

    # 划分训练集数据
    train_data_path = os.path.join(train_data_input_path, 'image')
    train_label_path = os.path.join(train_data_input_path, 'label')
    
    data_folders = sorted([x for x in os.listdir(train_data_path)])

    for data_folder in data_folders[:28]:
        data_source = os.path.join(train_data_path, data_folder)
        seg_source = os.path.join(train_label_path, f'label-{data_folder[5:]}')

        data_target = train_images_path
        seg_target = train_labels_path

        os.makedirs(data_target, exist_ok=True)
        os.makedirs(seg_target, exist_ok=True)

        patches = os.listdir(data_source)
        random.shuffle(patches)
        #selected_patches = patches[:20]
        selected_patches = patches

        for patch in selected_patches:
            patch_source = os.path.join(data_source, patch)
            seg_patch_source = os.path.join(seg_source, f'label-{patch[5:]}')

            patch_target = os.path.join(data_target, patch)
            seg_patch_target = os.path.join(seg_target, f'label-{patch[5:]}')

            shutil.copy(patch_source, patch_target)
            shutil.copy(seg_patch_source, seg_patch_target)
   
    train_indices = range(1, 29)
    for data_folder_id in train_indices:
        data_folder="data-"+str(data_folder_id)
        data_source = os.path.join(train_data_path, data_folder)
        seg_source = os.path.join(train_label_path, f'label-{data_folder[5:]}')

        # 注意：这里的目标路径改为验证集的路径
        data_target = train_images_path
        seg_target = train_labels_path

        os.makedirs(data_target, exist_ok=True)
        os.makedirs(seg_target, exist_ok=True)

        patches = os.listdir(data_source)
        random.shuffle(patches)
        selected_patches = patches  # 选择所有文件

        for patch in selected_patches:
            patch_source = os.path.join(data_source, patch)
            seg_patch_source = os.path.join(seg_source, f'label-{patch[5:]}')

            patch_target = os.path.join(data_target, patch)
            seg_patch_target = os.path.join(seg_target, f'label-{patch[5:]}')

            shutil.copy(patch_source, patch_target)
            shutil.copy(seg_patch_source, seg_patch_target)
    # 划分测试集数据
    test_data_source = os.path.join(test_data_input_path, 'image')
    test_label_source = os.path.join(test_data_input_path, 'label')

    test_indices = range(29, 37)  # 后20个数据

    for index in test_indices:
        data_folder = f'data-{index:02d}.nii.gz'
        seg_folder = f'label-{index:02d}.nii.gz'

        data_source = os.path.join(test_data_source, data_folder)
        seg_source = os.path.join(test_label_source, seg_folder)

        data_target = os.path.join(test_images_path, data_folder)
        seg_target = os.path.join(test_labels_path, seg_folder)

        shutil.copy(data_source, data_target)
        shutil.copy(seg_source, seg_target)

    print("Dataset splitting completed.")

''''''
def split_dataset(train_data_input_path, test_data_input_path):
    random.seed(42)     ## 设置随机数种子

    # 设置路径
    train_images_path = '/home/segmamba/CAS2023_data_preprocess/train_128_2000/data'
    train_labels_path = '/home/segmamba/CAS2023_data_preprocess/train_128_2000/mask'
    test_images_path = '/home/segmamba/CAS2023_data_preprocess/val_128_2000/data'
    test_labels_path = '/home/segmamba/CAS2023_data_preprocess/val_128_2000/mask'

    # 创建输出文件夹
    os.makedirs(train_images_path, exist_ok=True)
    os.makedirs(train_labels_path, exist_ok=True)
    os.makedirs(test_images_path, exist_ok=True)
    os.makedirs(test_labels_path, exist_ok=True)

    # 划分训练集数据
    train_data_path = os.path.join(train_data_input_path, 'data')
    train_label_path = os.path.join(train_data_input_path, 'mask')

    data_folders = sorted([x for x in os.listdir(train_data_path)])

    for data_folder in data_folders[:80]:
        data_source = os.path.join(train_data_path, data_folder)
        #seg_source = os.path.join(train_label_path, f'label-{data_folder[5:]}')
        #seg_source = os.path.join(train_label_path, data_folder+"_GT")
        seg_source = os.path.join(train_label_path, data_folder)
        data_target = train_images_path
        seg_target = train_labels_path

        os.makedirs(data_target, exist_ok=True)
        os.makedirs(seg_target, exist_ok=True)

        patches = os.listdir(data_source)
        random.shuffle(patches)
        #selected_patches = patches[:20]
        selected_patches = patches

        for patch in selected_patches:


            patch_source = os.path.join(data_source, patch)
            seg_patch_source = os.path.join(seg_source, patch)

            patch_target = os.path.join(data_target, patch)
            seg_patch_target = os.path.join(seg_target,patch)

            shutil.copy(patch_source, patch_target)
            shutil.copy(seg_patch_source, seg_patch_target)

    # 划分测试集数据
    test_data_source = os.path.join(test_data_input_path, 'data')
    test_label_source = os.path.join(test_data_input_path, 'mask')

    data_test = sorted([x for x in os.listdir(test_data_source)])

    for data_file in data_test[-20:]:
        #data_file_name = data_file.split('.')[0]
        data_source = os.path.join(test_data_source, data_file)
        seg_source= os.path.join(test_label_source, data_file)


        data_target = os.path.join(test_images_path, data_file)
        seg_target = os.path.join(test_labels_path, data_file)

        shutil.copy(data_source, data_target)
        shutil.copy(seg_source, seg_target)

    print("Dataset splitting completed.")



if __name__ == '__main__':

    # raw_data_path = '/media/ssd1/ly/Vessel/Datasets/WenYi/raw_data'      ## --- raw data 路径, 路径下目录为  WenYi/images ， WenYi/labels --- ##
    # preprocess_save_path = '/media/ssd1/ly/Vessel/Datasets/Public/preprocess_data/'

    # get_image_info(raw_data_path)           ## --- 先获取 raw data的信息， 方便后续比较 ---##
    # get_histograms(raw_data_path)           ## --- 先获取 raw_data的直方图信息 --- ##

    print("----- start data preprocess -----")
    # ## --- 第一步， 0背景去除 --- ##
    #remove_zero_background_output_path = remove_zero_background(raw_data_path)      ## --- 0背景去除 --- ##
    # get_image_info(remove_zero_background_output_path)                              ## --- 获取去除 0背景之后的 data 信息 ---##
    # #remove_zero_background_output_path = '/media/ssd1/ly/Vessel/Datasets/WenYi/preprocess_data/01_zero_background_removed'
    # get_histograms(remove_zero_background_output_path)                              ## --- 去除 0背景之后的 后直方图统计 ---##


    # ## --- 重采样 根据数据的space 确定重采样后的 space
    # reasmple_input_path = '/media/ssd1/ly/Vessel/Datasets/WenYi/raw_data/01_zero_background_removed'
    # reasmple(reasmple_input_path, out_space)


    ## --- 第二步，minmax归一化  /  zscore标准化 --- ##
    #normalize_input_path ="/home/segmamba/CAS2023_trainingdataset/"
    #normalization = "minmax"                                                                            ## ---归一化/标准化方式     zscore / minmax --- ##
    #normalize_output_path = normalize(normalize_input_path, normalization)                               ## --- minmax归一化  /  zscore标准化 --- ##
    #get_image_info(normalize_output_path)                                                               ## --- 获取去除 0背景之后的 data 信息 ---##
    #get_histograms(normalize_output_path)                                                               ## --- 归一化 / 标准化 后直方图统计 ---##



    ##--- 暂定 是否用代码划分 train 和 test 数据 --- ##

    split_patch_input_path = "/home/segmamba/CTAall/train"  ## --- normalize_output_path
    #--- "constant"（常数填充）、"edge"（边缘填充）、"reflect"（反射填充）和"symmetric"（对称反射填充）
    split_patch_out_path = split_patch(split_patch_input_path, HW_stride=96, D_stride=96, patch_size=(128, 128, 128), pad_way = 'constant')
    #get_image_info(split_patch_out_path)


    ## --- 根据label中包含血管的像素筛选patch --- ##
    #filter_patches_input_patch = '/home/segmamba/CAS2023_then_split/split_patch_pad_with_constant/64_64_size128/'

    #threshold = 2000
    #filter_patches(filter_patches_input_patch, threshold)
    print(" ----- finish data preprocess -----")


    ## --- 使用代码划分train集和test集 --- ##
    #train_data_input_path = '/home/segmamba/CAS2023_filter_threshold_2000/'
    #test_data_input_path = '/home/segmamba/CAS2023_normalized_zscore/'
    #split_dataset(train_data_input_path, test_data_input_path)






   



        
        

