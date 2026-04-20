import math
import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch.autograd import Function


# ====================== Triton 核函数 (3D 版本) ======================

@triton.jit
def _idx_3d(i, n: int, c: int, d: int, h: int, w: int):
    """将线性索引i解包为(n, c, d, h, w)坐标"""
    ni = i // (c * d * h * w)
    ci = (i // (d * h * w)) % c
    di = (i // (h * w)) % d
    hi = (i // w) % h
    wi = i % w
    m = i < (n * c * d * h * w)
    return ni, ci, di, hi, wi, m


@triton.jit
def ska_3d_fwd(
        x_ptr, w_ptr, o_ptr,  # 输入、权重、输出指针
        n, ic, d, h, w, ks, pad, wc,  # 新增深度维度 d
        BS: tl.constexpr,
        CT: tl.constexpr, AT: tl.constexpr
):
    pid = tl.program_id(0)
    start = pid * BS
    offs = start + tl.arange(0, BS)

    ni, ci, di, hi, wi, m = _idx_3d(offs, n, ic, d, h, w)
    val = tl.zeros((BS,), dtype=AT)

    for kd in range(ks):
        din = di - pad + kd
        db = (din >= 0) & (din < d)
        for kh in range(ks):
            hin = hi - pad + kh
            hb = db & (hin >= 0) & (hin < h)
            for kw in range(ks):
                win = wi - pad + kw
                b = hb & (win >= 0) & (win < w)

                x_off = (((ni * ic + ci) * d + din) * h + hin) * w + win
                w_off = (((ni * wc + ci % wc) * ks * ks * ks +
                          (kd * ks * ks + kh * ks + kw)) * d * h * w +
                         (di * h * w + hi * w + wi))

                x_val = tl.load(x_ptr + x_off, mask=m & b, other=0.0).to(CT)
                w_val = tl.load(w_ptr + w_off, mask=m, other=0.0).to(CT)
                val += tl.where(b & m, x_val * w_val, 0.0).to(AT)

    tl.store(o_ptr + offs, val.to(CT), mask=m)


@triton.jit
def ska_3d_bwd_x(
        go_ptr, w_ptr, gi_ptr,  # 梯度输出、权重、输入梯度指针
        n, ic, d, h, w, ks, pad, wc,  # 新增深度维度 d
        BS: tl.constexpr,
        CT: tl.constexpr, AT: tl.constexpr
):
    pid = tl.program_id(0)
    start = pid * BS
    offs = start + tl.arange(0, BS)

    ni, ci, di, hi, wi, m = _idx_3d(offs, n, ic, d, h, w)
    val = tl.zeros((BS,), dtype=AT)

    for kd in range(ks):
        do = di + pad - kd
        db = (do >= 0) & (do < d)
        for kh in range(ks):
            ho = hi + pad - kh
            hb = db & (ho >= 0) & (ho < h)
            for kw in range(ks):
                wo = wi + pad - kw
                b = hb & (wo >= 0) & (wo < w)

                go_off = (((ni * ic + ci) * d + do) * h + ho) * w + wo
                w_off = (((ni * wc + ci % wc) * ks * ks * ks +
                          (kd * ks * ks + kh * ks + kw)) * d * h * w +
                         (do * h * w + ho * w + wo))

                go_val = tl.load(go_ptr + go_off, mask=m & b, other=0.0).to(CT)
                w_val = tl.load(w_ptr + w_off, mask=m, other=0.0).to(CT)
                val += tl.where(b & m, go_val * w_val, 0.0).to(AT)

    tl.store(gi_ptr + offs, val.to(CT), mask=m)


@triton.jit
def ska_3d_bwd_w(
        go_ptr, x_ptr, gw_ptr,  # 梯度输出、输入、权重梯度指针
        n, wc, d, h, w, ic, ks, pad,  # 新增深度维度 d
        BS: tl.constexpr,
        CT: tl.constexpr, AT: tl.constexpr
):
    pid = tl.program_id(0)
    start = pid * BS
    offs = start + tl.arange(0, BS)

    ni, ci, di, hi, wi, m = _idx_3d(offs, n, wc, d, h, w)

    for kd in range(ks):
        din = di - pad + kd
        db = (din >= 0) & (din < d)
        for kh in range(ks):
            hin = hi - pad + kh
            hb = db & (hin >= 0) & (hin < h)
            for kw in range(ks):
                win = wi - pad + kw
                b = hb & (win >= 0) & (win < w)

                w_off = (((ni * wc + ci) * ks * ks * ks +
                          (kd * ks * ks + kh * ks + kw)) * d * h * w +
                         (di * h * w + hi * w + wi))

                val = tl.zeros((BS,), dtype=AT)
                steps = (ic - ci + wc - 1) // wc

                for s in range(tl.max(steps, axis=0)):
                    cc = ci + s * wc
                    cm = (cc < ic) & m & b

                    x_off = (((ni * ic + cc) * d + din) * h + hin) * w + win
                    go_off = (((ni * ic + cc) * d + di) * h + hi) * w + wi

                    x_val = tl.load(x_ptr + x_off, mask=cm, other=0.0).to(CT)
                    go_val = tl.load(go_ptr + go_off, mask=cm, other=0.0).to(CT)
                    val += tl.where(cm, x_val * go_val, 0.0).to(AT)

                tl.store(gw_ptr + w_off, val.to(CT), mask=m)


# ====================== PyTorch 函数封装 (3D) ======================

class SkaFn3D(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        ks = int(round(w.shape[2] ** (1 / 3)))  # 立方根计算
        pad = (ks - 1) // 2
        ctx.ks, ctx.pad = ks, pad
        n, ic, depth, height, width = x.shape
        wc = w.shape[1]

        o = torch.empty(n, ic, depth, height, width, device=x.device, dtype=x.dtype)
        numel = o.numel()

        # 确保内存连续
        x, w = x.contiguous(), w.contiguous()

        # 计算网格
        grid = lambda meta: (triton.cdiv(numel, meta['BS']),)

        # 数据类型处理
        ct = tl.float16 if x.dtype == torch.float16 else (
            tl.float32 if x.dtype == torch.float32 else tl.float64)
        at = tl.float32 if x.dtype == torch.float16 else ct

        # 调用核函数
        ska_3d_fwd[grid](
            x, w, o, n, ic, depth, height, width, ks, pad, wc,
            BS=128, CT=ct, AT=at
        )

        ctx.save_for_backward(x, w)
        ctx.ct, ctx.at = ct, at
        return o

    @staticmethod
    def backward(ctx, go: torch.Tensor):
        ks, pad = ctx.ks, ctx.pad
        x, w = ctx.saved_tensors
        n, ic, depth, height, width = x.shape
        wc = w.shape[1]
        ct, at = ctx.ct, ctx.at

        go = go.contiguous()
        gx = gw = None

        # 计算输入梯度
        if ctx.needs_input_grad[0]:
            gx = torch.empty_like(x)
            numel = gx.numel()
            grid = lambda meta: (triton.cdiv(numel, meta['BS']),)
            ska_3d_bwd_x[grid](
                go, w, gx, n, ic, depth, height, width, ks, pad, wc,
                BS=128, CT=ct, AT=at
            )

        # 计算权重梯度
        if ctx.needs_input_grad[1]:
            gw = torch.empty_like(w)
            numel = gw.numel() // w.shape[2]
            grid = lambda meta: (triton.cdiv(numel, meta['BS']),)
            ska_3d_bwd_w[grid](
                go, x, gw, n, wc, depth, height, width, ic, ks, pad,
                BS=128, CT=ct, AT=at
            )

        return gx, gw


# ====================== 3D 卷积基础模块 ======================

class Conv3d_BN(nn.Module):
    """3D卷积 + BatchNorm + 激活"""

    def __init__(self, in_channels, out_channels, ks=1, pad=0, groups=1):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=ks, padding=pad,
            groups=groups, bias=False
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# ====================== 3D LKP 模块 ======================

class LKP_3D(nn.Module):
    def __init__(self, dim, lks, sks, groups):
        """
        3D 局部核预测模块

        参数:
            dim: 输入通道数
            lks: 大卷积核尺寸（用于特征提取）
            sks: 小卷积核尺寸（用于动态卷积）
            groups: 分组数（控制权重复用程度）
        """
        super().__init__()
        # 特征压缩路径
        self.cv1 = Conv3d_BN(dim, dim // 2, ks=1)
        self.cv2 = Conv3d_BN(
            dim // 2, dim // 2,
            ks=lks, pad=(lks - 1) // 2,
            groups=dim // 2  # 深度可分离卷积
        )
        self.cv3 = Conv3d_BN(dim // 2, dim // 2)

        # 动态核生成层
        self.cv4 = nn.Conv3d(
            dim // 2, sks ** 3 * dim // groups,
            kernel_size=1, bias=False
        )

        # 归一化层
        self.norm = nn.GroupNorm(
            num_groups=dim // groups,
            num_channels=sks ** 3 * dim // groups
        )

        # 参数记录
        self.sks = sks
        self.groups = groups
        self.dim = dim

    def forward(self, x):
        # 输入形状: [B, C, D, H, W]
        # 特征提取
        x = self.cv1(x)
        x = self.cv2(x)
        x = self.cv3(x)

        # 动态核生成
        w = self.cv4(x)
        w = self.norm(w)

        # 重塑为动态卷积核形状
        b, _, depth, height, width = w.size()
        w = w.view(
            b,
            self.dim // self.groups,  # 输出组数
            self.sks ** 3,  # 卷积核元素数 (k_d × k_h × k_w)
            depth, height, width  # 空间位置
        )
        return w


# ====================== 完整的 3D LSConv ======================

class LSConv3D(nn.Module):
    def __init__(self, dim, lks=5, sks=3, groups=8):
        """
        3D LSConv 模块

        参数:
            dim: 输入通道数
            lks: 大卷积核尺寸（默认5）
            sks: 动态卷积核尺寸（默认3）
            groups: 分组数（默认8）
        """
        super().__init__()
        # 局部核预测模块
        self.lkp = LKP_3D(dim, lks, sks, groups)

        # 动态卷积应用
        self.ska = SkaFn3D.apply

        # 标准化和残差
        self.bn = nn.BatchNorm3d(dim)

        # 参数记录
        self.lks = lks
        self.sks = sks
        self.groups = groups

    def forward(self, x):
        # 输入形状: [B, C, D, H, W]
        # 1. 生成动态卷积核
        dynamic_kernel = self.lkp(x)

        # 2. 应用动态卷积
        x_conv = self.ska(x, dynamic_kernel)

        # 3. 标准化和残差连接
        return self.bn(x_conv) + x


# ====================== 示例使用 ======================

if __name__ == "__main__":
    # 创建测试数据
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch, channels, depth, height, width = 2, 48, 64, 64, 64
    x = torch.randn(batch, channels, depth, height, width).to(device)

    # 创建并测试模块
    model = LSConv3D(dim=channels).to(device)

    # 前向测试
    with torch.no_grad():
        y = model(x)
        print(f"输入形状: {tuple(x.shape)}")
        print(f"输出形状: {tuple(y.shape)}")

    # 反向测试 (需要梯度时)
    x.requires_grad = True
    y = model(x)
    loss = y.sum()
    loss.backward()
    print(f"输入梯度形状: {tuple(x.grad.shape)}")