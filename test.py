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
from networks.MSnet import MSUnet
from test_util import test_all_case

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='/home/segmamba', help='Name of Experiment')
    parser.add_argument('--model', type=str, default='Edge_Scan_SegMamba_v3_mean_edge_GSLSC_version2', help='model_name')
    parser.add_argument('--patch_size', type=list, default=[128, 128, 128], help='patch size of network input')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
    FLAGS = parser.parse_args()

    
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    

   
    proc_title = "zjx_deep_test"
    setproctitle.setproctitle(proc_title)

    
    torch.cuda.empty_cache()

    
    snapshot_path = f"/home/segmamba/model/{FLAGS.model}/"
    test_save_path = f"/home/segmamba/model/{FLAGS.model}/Prediction/"

    
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)

    
    if FLAGS.model == "SegMamba":
        
        net = SegMamba(
            in_chans=1,
            out_chans=2,
            depths=[1, 1, 1, 1],
            feat_size=[48, 96, 192, 384]
        )
    elif FLAGS.model == "unet":
        
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
    elif FLAGS.model == "MSUnet":
        
        net = MSUnet(n_channels=1, n_classes=2, n_filters=24)

    elif FLAGS.model == "UNETR":
        
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

    elif FLAGS.model == "Edge_Scan_SegMamba_v3_mean_edge":
        
        net = EdgeAwareScanSegMamba_v3(in_chans=1,
                                         out_chans=2,
                                         depths=[1, 1, 1, 1],
                                         feat_size=[48, 96, 192, 384])

    elif FLAGS.model == "Edge_Scan_SegMamba_v3_mean_edge_LSGC":
        
        net = EdgeAwareScanSegMamba_v3_GSLSC_version2(in_chans=1,
                                                      out_chans=2,
                                                      depths=[1, 1, 1, 1],
                                                      feat_size=[48, 96, 192, 384])
    elif FLAGS.model == "UXNet":
        
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
        
        net = AttentionUnet(
        spatial_dims=3,
        in_channels=1, 
        out_channels=2,
        channels=[48, 96, 192, 384],
        strides=[2, 2, 2, 2],
        )
    elif FLAGS.model == "MedNeXt":
       
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
        raise ValueError(f"Unidentified Model: {FLAGS.model}")

    
    save_mode_path = os.path.join(snapshot_path, f'{FLAGS.model}_best_model.pth')
    net.load_state_dict(torch.load(save_mode_path))
    print(f"Loading weight: {save_mode_path}")


    net = net.cuda()
    net.eval()

    

    with torch.no_grad():
        
        avg_metric = test_all_case(
            net=net,
            test_path='/home/segmamba/MRA_data_preprocess/val_128_2000/',
            num_classes=2,
            patch_size=(128, 128, 128),
            stride_xy=64,
            stride_z=64,
            test_save_path=test_save_path)


   
    print(f"Model: {FLAGS.model}, Dice: {avg_metric}")

   
    del net
    torch.cuda.empty_cache()

    print(f"Result - Model: {FLAGS.model}, Dice: {avg_metric}")
