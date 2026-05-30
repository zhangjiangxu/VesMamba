import argparse
import os
import shutil
import time
import torch
import setproctitle
from model_segmamba.segmamba import SegMamba
from monai.networks.nets.unet import UNet
from monai.networks.nets.attentionunet import AttentionUnet
from monai.networks.nets.unetr import UNETR
from monai.networks.nets.swin_unetr import SwinUNETR
from networks.vnet import VNet
from networks.Edge_Scan_Segmamba_v1 import EdgeAwareScanSegMamba_v1
from networks.Edge_Scan_Segmamba_v2 import EdgeAwareScanSegMamba_v2
from networks.Edge_Scan_Segmamba_v3 import EdgeAwareScanSegMamba_v3
from networks.UX_Net.network.UXNet_3D.network_backbone import UXNET
from MedNeXt.nnunet_mednext.network_architecture.mednextv1.MedNextV1 import MedNeXt
from networks.Edge_Scan_Segmamba_v3_GSLSConv import EdgeAwareScanSegMamba_v3_GSLSC
from networks.Edge_Scan_Segmamba_v3_GSLSConv_version2 import EdgeAwareScanSegMamba_v3_GSLSC_version2
from test_util import test_all_case

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='/home/segmamba', help='Name of Experiment')
    parser.add_argument('--model', type=str, default='Edge_Scan_SegMamba_v3_mean_edge_GSLSC_version2', help='model_name')
    parser.add_argument('--patch_size', type=list, default=[128, 128, 128], help='patch size of network input')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
    FLAGS = parser.parse_args()

   
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    print(f"使用 GPU: {FLAGS.gpu}")

   
    proc_title = "zjx_deep_test"
    setproctitle.setproctitle(proc_title)

    
    torch.cuda.empty_cache()

    snapshot_path = f"/home/segmamba/model_/{FLAGS.model}/"
    test_save_path = f"/home/segmamba/model/{FLAGS.model}/Prediction/"

    
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)


    if FLAGS.model == "SegMamba":
        print('使用 SegMamba 模型')
        net = SegMamba(
            in_chans=1,
            out_chans=2,
            depths=[1, 1, 1, 1],
            feat_size=[48, 96, 192, 384]
        )
    elif FLAGS.model == "unet":
        print('使用 UNet 模型')
        net = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(48, 96, 192, 384, 768),
            strides=(2, 2, 2, 2),
            num_res_units=0,
            kernel_size=3,
            up_kernel_size=3,
            norm="instance",
            act="relu"
        )
   
    elif FLAGS.model == "UNETR":
        print('使用 UNETR 模型')
        net = UNETR(
            in_channels=1,
            out_channels=2,
            img_size=(128, 128, 128),
            feature_size=48,
            hidden_size=384,
            mlp_dim=768,
            num_heads=6,
            norm_name='instance',
            conv_block=True,
            res_block=True,
            dropout_rate=0.0,
            spatial_dims=3
        )
    elif FLAGS.model == "Swin_UNETR":
        print('使用 Swin_UNETR 模型')
        net = SwinUNETR(
            img_size=(128, 128, 128),  
            in_channels=1,  
            out_channels=2,  
            feature_size=48,  
            depths=(1, 1, 1, 1), 
            num_heads=(6, 6, 6, 6),  
            norm_name='instance',  
            drop_rate=0.0,  
            attn_drop_rate=0.0,  
            dropout_path_rate=0.0,  
            use_checkpoint=False,  
            spatial_dims=3,  
            downsample="merging",  
            use_v2=False 
            )

    elif FLAGS.model == "Edge_Scan_SegMamba_v1_mean_edge":
        print('使用 Edge_Scan_SegMamba_v1_mean_edge 模型')
        net = EdgeAwareScanSegMamba_v1(in_chans=1,
                                         out_chans=2,
                                         depths=[1, 1, 1, 1],
                                         feat_size=[48, 96, 192, 384])
    elif FLAGS.model == "Edge_Scan_SegMamba_v2_mean_edge":
        print('使用 Edge_Scan_SegMamba_v2_mean_edge 模型')
        net = EdgeAwareScanSegMamba_v2(in_chans=1,
                                         out_chans=2,
                                         depths=[1, 1, 1, 1],
                                         feat_size=[48, 96, 192, 384])
    elif FLAGS.model == "Edge_Scan_SegMamba_v3_mean_edge":
        print('使用 Edge_Scan_SegMamba_v3_mean_edge 模型')
        net = EdgeAwareScanSegMamba_v3(in_chans=1,
                                         out_chans=2,
                                         depths=[1, 1, 1, 1],
                                         feat_size=[48, 96, 192, 384])
    elif FLAGS.model == "Edge_Scan_SegMamba_v3_mean_edge_GSLSC":
        print('使用 Edge_Scan_SegMamba_v3_mean_edge_GSLSC 模型')
        net = EdgeAwareScanSegMamba_v3_GSLSC(in_chans=1,
                                            out_chans=2,
                                            depths=[1, 1, 1, 1],
                                            feat_size=[48, 96, 192, 384])
    elif FLAGS.model == "Edge_Scan_SegMamba_v3_mean_edge_GSLSC_version2":
        print('使用 Edge_Scan_SegMamba_v3_mean_edge_GSLSC_version2 模型')
        net =  EdgeAwareScanSegMamba_v3_GSLSC_version2(in_chans=1,
                                            out_chans=2,
                                            depths=[1, 1, 1, 1],
                                            feat_size=[48, 96, 192, 384])
    elif FLAGS.model == "UXNet":
        print('使用 UXNet 模型')
        net = UXNET(in_chans=1,
        out_chans=2,
        depths=[1, 1, 1, 1],
        feat_size=[48, 96, 192, 384],
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        hidden_size = 768,
        norm_name= "instance",
        conv_block= True,
        res_block= True,
        spatial_dims=3)

    elif FLAGS.model == "attention_unet":
        print('使用 attention_unet 模型')
        net = AttentionUnet(
        spatial_dims=3,
        in_channels=1,  
        out_channels=2,
        channels=[48, 96, 192, 384],
        strides=[2, 2, 2, 2],
        )
    elif FLAGS.model == "MedNeXt":
        print('使用 MedNeXt 模型')
        net = MedNeXt(in_channels=1,
                        n_channels=48,
                        n_classes=2,
                        exp_r=2,
                        kernel_size=7,
                        block_counts=[1, 1, 1, 1, 1, 1, 1, 1, 1],
                        norm_type='group',
                        dim='3d',
                        )
    else:
        raise ValueError(f"未知模型: {FLAGS.model}")

    
    save_mode_path = os.path.join(snapshot_path, f'{FLAGS.model}_best_model.pth')
    net.load_state_dict(torch.load(save_mode_path))
    print(f"加载权重: {save_mode_path}")


    net = net.cuda()
    net.eval()

    
    start_time = time.time()
    with torch.no_grad():
       
        avg_metric = test_all_case(
            net=net,
            test_path='/home/segmamba/MRA_data_preprocess/val_128_2000/', 
            num_classes=2,                                                 
            patch_size=(128, 128, 128),
            stride_xy=64,
            stride_z=64,
            test_save_path=test_save_path)

    end_time = time.time()
    print(f"推理时间: {end_time - start_time:.2f}秒")
    print(f"模型: {FLAGS.model}, Dice系数: {avg_metric}")

   
    del net
    torch.cuda.empty_cache()

    print(f"最终结果 - 模型: {FLAGS.model}, Dice系数: {avg_metric}")
