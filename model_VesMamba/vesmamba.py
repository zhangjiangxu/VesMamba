# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
import torch.nn as nn
import torch 
from functools import partial

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from .mamba_vessel import Mamba
import torch.nn.functional as F
import torch.nn.init as init
#from .Vascular_Sobel import  EnhancedEdgeAwareBlock
import sys
from lsnet.lsconv_3D import  LSConv3D

class VascularSobel3D(nn.Module):
    """优化的3D Sobel边缘检测器 (仅使用3×3×3核)

    专为3D血管分割设计，使用固定权重的3D Sobel算子提取边缘特征，
    保留每个通道的空间独立性。

    输入: [B, C, D, H, W] (批次大小, 通道数, 深度, 高度, 宽度)
    输出: [B, C, D, H, W] (边缘强度图，值域[0,1])
    """

    def __init__(self):
        """
        初始化3D Sobel边缘检测器
        """
        super().__init__()
        # 初始化三个方向的Sobel核
        self._init_3x3_kernels()

    def _init_3x3_kernels(self):
        """构造并注册三个方向的3×3×3 Sobel核"""

        # X方向(宽度方向)核: 检测左右边缘 [1,1,3,3,3]
        sobel_x = torch.tensor([
            [[[1, 0, -1], [2, 0, -2], [1, 0, -1]],
             [[2, 0, -2], [4, 0, -4], [2, 0, -2]],
             [[1, 0, -1], [2, 0, -2], [1, 0, -1]]]
        ], dtype=torch.float32) / 8.0  # 归一化

        # Y方向(高度方向)核: 检测上下边缘 (X核的转置)
        sobel_y = sobel_x.transpose(2, 3)

        # Z方向(深度方向)核: 检测前后边缘
        sobel_z = torch.tensor([
            [[[1, 2, 1], [2, 4, 2], [1, 2, 1]],
             [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
             [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]]]
        ], dtype=torch.float32) / 8.0  # 归一化

        # 注册为不参与训练的buffer
        self.register_buffer('kernel_x', sobel_x.view(1, 1, 3, 3, 3))
        self.register_buffer('kernel_y', sobel_y.view(1, 1, 3, 3, 3))
        self.register_buffer('kernel_z', sobel_z.view(1, 1, 3, 3, 3))

    def forward(self, x):
        """
        计算输入体积的边缘强度

        参数:
            x: 输入张量 [B, C, D, H, W]
        返回:
            边缘强度图 [B, C, D, H, W]，归一化到[0,1]范围
        """
        B, C, D, H, W = x.shape
        pad = 1  # 维持空间尺寸不变的padding值

        # 对每个通道单独计算三个方向的梯度
        grad_x = F.conv3d(x, self.kernel_x.repeat(C, 1, 1, 1, 1),
                          padding=pad, groups=C)  # X方向梯度
        grad_y = F.conv3d(x, self.kernel_y.repeat(C, 1, 1, 1, 1),
                          padding=pad, groups=C)  # Y方向梯度
        grad_z = F.conv3d(x, self.kernel_z.repeat(C, 1, 1, 1, 1),
                          padding=pad, groups=C)  # Z方向梯度

        # 计算梯度幅值作为边缘强度
        edge = torch.sqrt(grad_x ** 2 + grad_y ** 2 + grad_z ** 2 + 1e-6)  # 加1e-6避免除零

        # 通道独立归一化: 各通道除以其最大值
        return edge / (edge.amax(dim=(2, 3, 4), keepdim=True) + 1e-6)


class EnhancedEdgeAwareBlock(nn.Module):
    """血管分割专用边缘感知模块 (3×3×3)

    核心功能:
    1. 使用3D Sobel算子提取血管边缘特征
    2. 通道注意力: 增强重要特征通道
    3. 空间注意力: 聚焦血管边缘区域
    4. 特征融合: 整合边缘信息到原始特征

    输入: [B, C, D, H, W] (批次大小, 通道数, 深度, 高度, 宽度)
    输出: [B, C, D, H, W] (增强后的边缘特征图)
    """

    def __init__(self, in_channels):
        """
        初始化边缘感知模块

        参数:
            in_channels: 输入特征图的通道数
        """
        super().__init__()

        # 3D边缘检测器 (仅3×3×3核)
        self.edge_detector = VascularSobel3D()

        # 通道注意力机制 (学习通道重要性权重)
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),  # 全局平均池化: [B, C, 1, 1, 1]
            nn.Conv3d(in_channels, max(4, in_channels // 8), 1),  # 降维压缩
            nn.ReLU(inplace=True),  # 非线性激活
            nn.Conv3d(max(4, in_channels // 8), in_channels, 1),  # 恢复通道数
            nn.Sigmoid()  # 输出0-1通道权重
        )

        # 空间注意力机制 (学习空间重要性权重)
        self.spatial_att = nn.Sequential(
            nn.Conv3d(1, 4, kernel_size=7, padding=3),  # 大核捕获空间上下文
            nn.ReLU(inplace=True),  # 非线性激活
            nn.Conv3d(4, 1, kernel_size=7, padding=3),  # 生成空间注意力图
            nn.Sigmoid()  # 输出0-1空间权重
        )

        # 特征融合卷积
        self.fusion_conv = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, 3, padding=1),  # 3D卷积融合特征
            nn.InstanceNorm3d(in_channels),  # 实例归一化(保留序列特征)
            nn.ReLU(inplace=True)  # 非线性激活
        )

    def forward(self, x):

        with torch.no_grad():
            # 提取原始空间边缘
            raw_edges = self.edge_detector(x)  # [B, C, D, H, W]
        return raw_edges


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    支持两种数据格式的LayerNorm：
    - channels_last: (batch_size, height, width, channels)
    - channels_first: (batch_size, channels, height, width)
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        """
        normalized_shape：要进行归一化的维度（通常是通道数）。
        eps=1e-6：防止除零的小数值，保证数值稳定。
        data_format="channels_last"：默认数据格式是 channels_last
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))  # 可学习的缩放参数
        self.bias = nn.Parameter(torch.zeros(normalized_shape))  # 可学习的偏置参数
        self.eps = eps  # 数值稳定性的小常数
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            # 使用PyTorch原生LayerNorm处理channels_last格式
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # 手动实现channels_first的归一化
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]

            return x


class Edge_scan_MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2,low_thresh=0.2, high_thresh=0.8, num_slices=None):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        #self.low_thresh = low_thresh
        #self.high_thresh = high_thresh

        self.TOM_mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
            bimamba_type="v3",
            nslices=num_slices,
        )

        self.Edge_mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
            bimamba_type="v3",
            nslices=num_slices,
        )

    def forward(self, x, edge_info=None):
        B, C, D, H, W = x.shape
        x_skip = x  # 残差连接
        N = D * H * W

        # 展平特征 [B, C, N] -> [B, N, C]
        x_flat = x.view(B, C, N).transpose(1, 2)  # [B, N, C]

        if edge_info is None:
            # 无边缘信息时处理原始序列
            x_out = self.TOM_mamba(self.norm(x_flat))
        else:
            assert edge_info.shape == (B, C, D, H, W)

            # 1. 展平边缘信息 [B, C, D, H, W] -> [B, C, N]
            edge_info_flat = edge_info.view(B, C, N)

            # 2. 计算全局边缘图 (所有通道平均)
            global_edge = torch.mean(edge_info_flat, dim=1, keepdim=True)  # [B, 1, N]

            # 3. 生成全局排序索引 (所有通道共享)
            sorted_indices = torch.argsort(global_edge, dim=2, descending=True)  # [B, 1, N]
            sorted_indices = sorted_indices.expand(-1, C, -1)  # 扩展到所有通道 [B, C, N]

            # 4. 应用排序到特征序列
            batch_idx = torch.arange(B, device=x.device)[:, None, None]  # [B, 1, 1]
            channel_idx = torch.arange(C, device=x.device)[None, :, None]  # [1, C, 1]
            sorted_x = torch.gather(x_flat, dim=1, index=sorted_indices.permute(0, 2, 1).expand(-1, -1, C))

            # 5. 边缘感知Mamba处理
            x_mamba1 = self.Edge_mamba(self.norm(sorted_x))

            # 6. 恢复原始顺序
            reverse_indices = torch.argsort(sorted_indices, dim=2)  # [B, C, N]升序012原顺序
            restored_x = torch.gather(
                x_mamba1,
                dim=1,
                index=reverse_indices.permute(0, 2, 1).expand(-1, -1, C)
            )

            #x_mamba2 = self.TOM_mamba(self.norm(x_flat))
            # 7. 融合双分支输出 (这里简化为单分支)
            x_out = restored_x

        # 恢复特征格式和空间维度
        x_out = x_out.transpose(1, 2).view(B, C, D, H, W)  # [B, C, N] -> [B, C, D, H, W]
        return x_out + x_skip


def encode_geometric_position(x_3d):
    """为每个体素添加几何坐标特征"""
    B, C, D, H, W = x_3d.shape
    # 生成归一化坐标网格
    z_coord = torch.linspace(-1, 1, D).view(1, 1, D, 1, 1).to(x_3d.device)
    y_coord = torch.linspace(-1, 1, H).view(1, 1, 1, H, 1).to(x_3d.device)
    x_coord = torch.linspace(-1, 1, W).view(1, 1, 1, 1, W).to(x_3d.device)

    # 拼接为几何特征
    geo_feat = torch.cat([
        z_coord.expand(B, 1, D, H, W),
        y_coord.expand(B, 1, D, H, W),
        x_coord.expand(B, 1, D, H, W)
    ], dim=1)     # [B,3,D,H,W]

    return geo_feat



class MlpChannel(nn.Module):
    def __init__(self, hidden_size, mlp_dim, ):
        super().__init__()
        self.fc1 = nn.Conv3d(hidden_size, mlp_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(mlp_dim, hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class GSC(nn.Module):
    def __init__(self, in_channles) -> None:
        super().__init__()

        self.proj = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
        self.norm = nn.InstanceNorm3d(in_channles)
        self.nonliner = nn.ReLU()

        self.proj2 = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
        self.norm2 = nn.InstanceNorm3d(in_channles)
        self.nonliner2 = nn.ReLU()

        self.proj3 = nn.Conv3d(in_channles, in_channles, 1, 1, 0)
        self.norm3 = nn.InstanceNorm3d(in_channles)
        self.nonliner3 = nn.ReLU()

        self.proj4 = nn.Conv3d(in_channles, in_channles, 1, 1, 0)
        self.norm4 = nn.InstanceNorm3d(in_channles)
        self.nonliner4 = nn.ReLU()

    def forward(self, x):
        x_residual = x

        x1 = self.proj(x)
        x1 = self.norm(x1)
        x1 = self.nonliner(x1)

        x1 = self.proj2(x1)
        x1 = self.norm2(x1)
        x1 = self.nonliner2(x1)

        x2 = self.proj3(x)
        x2 = self.norm3(x2)
        x2 = self.nonliner3(x2)

        x = x1 + x2
        x = self.proj4(x)
        x = self.norm4(x)
        x = self.nonliner4(x)

        return x + x_residual

class EnhancedGSC(nn.Module):
    def __init__(self, in_channels, lsconv_groups=8) -> None:
        """
            增强型 GSC 模块，集成 LSConv3D

            参数:
                in_channels: 输入通道数
                lsconv_groups: LSConv3D 的分组数（控制计算复杂度）
        """
        super().__init__()


        # 第二分支：点卷积路径
        self.pointwise_path = nn.Sequential(
                nn.Conv3d(in_channels, in_channels, 1, 1, 0),
                nn.InstanceNorm3d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(in_channels, in_channels, 1, 1, 0),
                nn.InstanceNorm3d(in_channels),
                nn.ReLU(inplace=True)
        )

        # 第三分支：LSConv3D 动态路径（新增）
        self.lsconv_path = nn.Sequential(
                LSConv3D(in_channels, groups=lsconv_groups),  # 动态卷积
                nn.InstanceNorm3d(in_channels),
                nn.ReLU(inplace=True)
        )


        # 最终输出层

        self.final_conv = nn.Conv3d(in_channels, in_channels, 1, 1, 0)
        self.final_norm = nn.InstanceNorm3d(in_channels)
        self.final_act = nn.ReLU(inplace=True)

    def forward(self, x):
        x_residual = x

        # 三个并行分支
        #x_conv = self.conv_path(x)
        x_point = self.pointwise_path(x)
        x_dynamic = self.lsconv_path(x)  # 动态特征提取

        # 特征融合

        x_fused = x_point + x_dynamic
        # 最终输出
        x_out = self.final_act(self.final_norm(self.final_conv(x_fused)))
        return x_out + x_residual





class EdgeAwareScanMambaEncoder(nn.Module):
    def __init__(self, in_chans=1, depths=[2, 2, 2, 2], dims=[48, 96, 192, 384],
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3]):  # CT灰度图像
        """
        in_chans	输入通道数（通常是1，比如灰度CT图像）
        depths	每个stage里面MambaLayer的个数，比如 [2,2,2,2] 表示每个stage有2个block
        dims	每个stage输出的特征通道数，比如[48, 96, 192, 384]
        drop_path_rate	随机深度(drop path)的比例（没用到）
        layer_scale_init_value	层归一化初始值（没用到）
        out_indices	哪几个stage输出（比如[0,1,2,3]表示全输出）
        """

        super().__init__()
        # 存放了stem和3个下采样块
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers

        # 初始化边缘感知模块（每个阶段一个）
        self.edge_blocks = nn.ModuleList([
            EnhancedEdgeAwareBlock(in_channels=dims[i])
            for i in range(len(dims))
        ])

        ### 起始的stem层
        stem = nn.Sequential(
            nn.Conv3d(in_chans, dims[0], kernel_size=7, stride=2, padding=3),  # 2,1,96,96,96 => 2,48,48,48,48
        )
        self.downsample_layers.append(stem)

        for i in range(3):  # i=0 1 2, dims=[48, 96, 192, 384]
            downsample_layer = nn.Sequential(  # 2,48,48,48,48 => 2,96,24,24,24
                # LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),      # 2,96,24,24,24 => 2,192,12,12,12
                nn.InstanceNorm3d(dims[i]),  # 对每通道归一化                          # 2,192,12,12,12 => 2,384,6,6,6
                nn.Conv3d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        self.gscs = nn.ModuleList()
        num_slices_list = [64, 32, 16, 8]
        cur = 0
        for i in range(4):
            #gsc = GSC(dims[i])
            # 根据特征图尺寸调整LSConv分组数（小特征图用更多组）
            groups = max(4, dims[i] // 16)  # 动态调整分组数
            gsc = EnhancedGSC(dims[i], lsconv_groups=groups)
            stage = nn.Sequential(
                *[Edge_scan_MambaLayer(dim=dims[i], num_slices=num_slices_list[i]) for j in range(depths[i])]
            )

            self.stages.append(stage)
            self.gscs.append(gsc)
            cur += depths[i]

        self.out_indices = out_indices

        self.mlps = nn.ModuleList()
        for i_layer in range(4):
            layer = nn.InstanceNorm3d(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
            self.mlps.append(MlpChannel(dims[i_layer], 2 * dims[i_layer]))

    def forward_features(self, x):
        outs = []  # 保存每层的输出特征

        for i in range(4):
            # print(f"Before downsample {i}: x.shape={x.shape}")

            # 1. 下采样
            x = self.downsample_layers[i](x)  # stem或下采样

            # 2. GSC处理（保持原结构）
            x = self.gscs[i](x)  # GSC

            edge_map = self.edge_blocks[i](x) # 3. 边缘特征提取（新增）

            # 4. Mamba处理（传入edge_map）
            ###x = self.stages[i](x)  # mamba
            for mamba_layer in self.stages[i]:
                x = mamba_layer(x, edge_map)

            if i in self.out_indices:  # 0 1 2 3
                norm_layer = getattr(self, f'norm{i}')  # 找到对应的归一化层 norm{i}（其实就是 InstanceNorm3d(dims[i])）
                x_out = norm_layer(x)  # 归一化
                x_out = self.mlps[i](x_out)  # MLP层
                outs.append(x_out)  # 输出添加outs

        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)
        return x

class  EdgeAwareScanSegMamba_v3_GSLSC_version2(nn.Module):
    def __init__(
            self,
            in_chans=1,
            out_chans=2,
            depths=[2, 2, 2, 2],
            feat_size=[48, 96, 192, 384],
            drop_path_rate=0,
            layer_scale_init_value=1e-6,
            hidden_size: int = 768,
            norm_name="instance",
            conv_block: bool = True,
            res_block: bool = True,
            spatial_dims=3,

    ) -> None:
        super().__init__()



        self.hidden_size = hidden_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.depths = depths
        self.drop_path_rate = drop_path_rate
        self.feat_size = feat_size
        self.layer_scale_init_value = layer_scale_init_value

        self.spatial_dims = spatial_dims
        self.vit = EdgeAwareScanMambaEncoder(in_chans,
                                depths=depths,
                                dims=feat_size,
                                drop_path_rate=drop_path_rate,
                                layer_scale_init_value=layer_scale_init_value,
                                )
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.in_chans,
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[2],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[3],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.hidden_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.hidden_size,
            out_channels=self.feat_size[3],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.feat_size[2],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[1],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=48, out_channels=self.out_chans)
        # self.apply(initialize_weights) ###初始化

    def proj_feat(self, x):
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x

    def forward(self, x_in):
        # print("x_in",x_in.shape)
        outs = self.vit(x_in)  # vit = MambaEncoder
        enc1 = self.encoder1(x_in)
        x2 = outs[0]
        enc2 = self.encoder2(x2)
        x3 = outs[1]
        enc3 = self.encoder3(x3)
        x4 = outs[2]
        enc4 = self.encoder4(x4)
        enc_hidden = self.encoder5(outs[3])
        dec3 = self.decoder5(enc_hidden, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0)

        return self.out(out)

if __name__=='__main__':
    # 1. 初始化网络
    '''
    net = EdgeAwareScanSegMamba_Edge_Dynamic_scan(in_chans=1,
                         out_chans=2,
                         depths=[2, 2, 2, 2],
                         feat_size=[48, 96, 192, 384]).cuda()  # 输入1通道，输出2通道（二分类）

    # 2. 生成随机输入数据（模拟3D医学图像）
    # 假设输入尺寸为 (batch_size=1, channels=1, depth=64, height=64, width=64)
    random_input = torch.rand(2, 1, 96, 96, 96).cuda()# 随机生成[0,1)之间的张量

    # 3. 前向传播测试
    output = net(random_input)

    # 4. 打印输出形状和值范围
    print(f"Input shape: {random_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output min/max: {output.min().item():.4f}, {output.max().item():.4f}")'''