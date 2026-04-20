import os
import csv
import math
import nibabel as nib
import numpy as np
from medpy import metric
import torch
import torch.nn.functional as F
from tqdm import tqdm
import SimpleITK as sitk
import time
'''
def test_all_case(net, test_path, num_classes, patch_size=(128, 128, 128), stride_xy=64, stride_z=64, save_result=True, test_save_path=None):
    total_metric = 0.0
    image_list = [f for f in os.listdir(os.path.join(test_path, 'data'))
                  if f.endswith(('.nii', '.nii.gz'))]
    for image_name in tqdm(image_list):
        print('   ')
        print('Processing:', image_name)
        image_path = os.path.join(test_path, "data", image_name)
        print("image_path:", image_path)
        label_path = os.path.join(test_path, "seg", image_name.replace("data", "seg"))
        print("label_path:", label_path)
        # 读取原始图像和标签（保留空间信息）
        original_image_sitk = sitk.ReadImage(image_path)
        original_label_sitk = sitk.ReadImage(label_path)

        # 获取图像的空间信息
        origin = original_image_sitk.GetOrigin()
        spacing = original_image_sitk.GetSpacing()
        direction = original_image_sitk.GetDirection()

        # 转换为numpy数组进行处理
        image = sitk.GetArrayFromImage(original_image_sitk).astype(np.float32)
        label = sitk.GetArrayFromImage(original_label_sitk).astype(np.float32)

        prediction, score_map = test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=num_classes)

        if np.sum(prediction)==0:
            single_metric = (0,0,0,0,0,0,0,0)
        else:
            single_metric = calculate_metric_percase(prediction, label)
        total_metric += np.asarray(single_metric)
        if save_result:
            # 从numpy数组创建SimpleITK图像
            pred_sitk = sitk.GetImageFromArray(prediction.astype(np.float32))

            # 设置空间信息
            pred_sitk.SetOrigin(origin)
            pred_sitk.SetSpacing(spacing)
            pred_sitk.SetDirection(direction)

            # 创建输出目录（如果不存在）
            os.makedirs(test_save_path, exist_ok=True)
            # 保存文件
            sitk.WriteImage(pred_sitk, os.path.join(test_save_path, image_name.replace("data", "pre")))
            print("Save result:", os.path.join(test_save_path, image_name.replace("data", "pre")))
        print("=======================================================================================================")
    avg_metric = total_metric / len(image_list)
    print('average metric is {}'.format(avg_metric))

    return avg_metric
'''


def test_all_case(net, test_path, num_classes, patch_size=(128, 128, 128),
                  stride_xy=64, stride_z=64, save_result=True, test_save_path=None):
    """
    测试所有案例并记录每个案例的指标到CSV文件

    参数:
        net: 神经网络模型
        test_path: 测试数据路径
        num_classes: 类别数
        patch_size: 输入块大小
        stride_xy: XY方向步长
        stride_z: Z方向步长
        save_result: 是否保存预测结果
        test_save_path: 预测结果保存路径
    """
    # 确保保存路径存在
    os.makedirs(test_save_path, exist_ok=True)

    # 创建CSV文件记录结果
    csv_path = os.path.join(test_save_path, "results.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # 写入表头 (移除了敏感性和特异性)
        csv_writer.writerow([
            "Image", "Dice", "JC", "Precision", "Recall",
            "HD", "HD95", "ASSD", "RAVD"
        ])

        # 获取所有测试图像
        image_list = [f for f in os.listdir(os.path.join(test_path, 'data'))
                      if f.endswith(('.nii', '.nii.gz'))]

        # 初始化总指标 (8个指标)
        total_metrics = np.zeros(8)
        num_images = len(image_list)

        for image_name in tqdm(image_list, desc="Processing images"):
            print(f'Processing: {image_name}')
            image_path = os.path.join(test_path, "data", image_name)
            label_path = os.path.join(test_path, "seg", image_name.replace("data", "seg"))

            # 读取图像和标签
            image_sitk = sitk.ReadImage(image_path)
            label_sitk = sitk.ReadImage(label_path)

            # 获取空间信息
            origin = image_sitk.GetOrigin()
            spacing = image_sitk.GetSpacing()
            direction = image_sitk.GetDirection()

            # 转换为numpy数组
            image_arr = sitk.GetArrayFromImage(image_sitk).astype(np.float32)
            label_arr = sitk.GetArrayFromImage(label_sitk).astype(np.float32)

            # 进行预测
            time1=time.time()
            prediction, _ = test_single_case(
                net, image_arr, stride_xy, stride_z, patch_size, num_classes
            )
            time2=time.time()
            time_cost=time2-time1
            print("one case time_cost:", time_cost)
            break


            # 计算指标 (移除了敏感性和特异性)
            if np.sum(prediction) == 0:
                metrics = (0, 0, 0, 0, 0, 0, 0, 0)
            else:
                metrics = calculate_metric_percase(prediction, label_arr)

            # 记录指标到CSV
            csv_writer.writerow([image_name] + list(metrics))

            # 累加指标
            total_metrics += np.array(metrics)

            # 保存预测结果
            if save_result:
                pred_sitk = sitk.GetImageFromArray(prediction.astype(np.float32))
                pred_sitk.SetOrigin(origin)
                pred_sitk.SetSpacing(spacing)
                pred_sitk.SetDirection(direction)

                pred_path = os.path.join(test_save_path, image_name.replace("data", "pred"))
                sitk.WriteImage(pred_sitk, pred_path)
                print(f"Saved prediction: {pred_path}")

            print("===============================")

        # 计算平均指标
        avg_metrics = total_metrics / num_images
        print(f'Average metrics: {avg_metrics}')

        # 写入平均指标
        csv_writer.writerow(["Average"] + list(avg_metrics))

    return avg_metrics

def test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=2):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2,w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2,h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2,d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(wl_pad,wr_pad),(hl_pad,hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww,hh,dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    #print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y,hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch,axis=0),axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()
                y1 = net(test_patch)
                y = F.softmax(y1, dim=1)
                y = y.cpu().data.numpy()
                y = y[0,:,:,:,:]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
    score_map = score_map/np.expand_dims(cnt,axis=0)
    label_map = np.argmax(score_map, axis = 0)
    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
        score_map = score_map[:,wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
    return label_map, score_map


def calculate_metric_percase(pred, gt):
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    precision = metric.binary.precision(pred, gt)
    recall = metric.binary.recall(pred, gt)
    hd = metric.binary.hd(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)
    assd = metric.binary.assd(pred, gt)
    ravd = metric.binary.ravd(pred, gt)
    return dice, jc, precision, recall, hd, hd95, assd, ravd
