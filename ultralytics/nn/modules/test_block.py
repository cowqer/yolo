# import pywt
# import pywt.data
# import torch
# from torch import nn
# from functools import partial
# import torch.nn.functional as F


# # 论文地址 https://arxiv.org/pdf/2407.05848
# def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
#     w = pywt.Wavelet(wave)
#     dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
#     dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
#     dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
#                                dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
#                                dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
#                                dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

#     dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

#     rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
#     rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
#     rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
#                                rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
#                                rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
#                                rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

#     rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

#     return dec_filters, rec_filters


# def wavelet_transform(x, filters):
#     b, c, h, w = x.shape
#     pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
#     print("Filters Shape:", filters.shape)  # Should match the expected input channels
#     print("Input Shape:", x.shape)  # Should have the number of channels that matches the filters

#     x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
#     print("Before reshape:", x.shape)
#     x = x.reshape(b, c, 4, h // 2, w // 2)
#     return x


# def inverse_wavelet_transform(x, filters):
#     b, c, _, h_half, w_half = x.shape
#     pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
#     x = x.reshape(b, c * 4, h_half, w_half)
#     x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
#     return x


# # Wavelet Transform Conv(WTConv2d)
# class WTConv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, bias=True, wt_levels=1, wt_type='db1'):
#         super(WTConv2d, self).__init__()

#         assert in_channels == out_channels

#         self.in_channels = in_channels
#         self.wt_levels = wt_levels
#         self.stride = stride
#         self.dilation = 1

#         self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
#         self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
#         self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

#         self.wt_function = partial(wavelet_transform, filters=self.wt_filter)
#         self.iwt_function = partial(inverse_wavelet_transform, filters=self.iwt_filter)

#         self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', stride=1, dilation=1,
#                                    groups=in_channels, bias=bias)
#         self.base_scale = _ScaleModule([1, in_channels, 1, 1])

#         self.wavelet_convs = nn.ModuleList(
#             [nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size, padding='same', stride=1, dilation=1,
#                        groups=in_channels * 4, bias=False) for _ in range(self.wt_levels)]
#         )
#         self.wavelet_scale = nn.ModuleList(
#             [_ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)]
#         )

#         if self.stride > 1:
#             self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=False)
#             self.do_stride = lambda x_in: F.conv2d(x_in, self.stride_filter, bias=None, stride=self.stride,
#                                                    groups=in_channels)
#         else:
#             self.do_stride = None

#     def forward(self, x):

#         x_ll_in_levels = []
#         x_h_in_levels = []
#         shapes_in_levels = []

#         curr_x_ll = x

#         for i in range(self.wt_levels):
#             curr_shape = curr_x_ll.shape
#             shapes_in_levels.append(curr_shape)
#             if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
#                 curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
#                 curr_x_ll = F.pad(curr_x_ll, curr_pads)

#             curr_x = self.wt_function(curr_x_ll)
#             curr_x_ll = curr_x[:, :, 0, :, :]

#             shape_x = curr_x.shape
#             curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
#             curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
#             curr_x_tag = curr_x_tag.reshape(shape_x)

#             x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
#             x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

#         next_x_ll = 0

#         for i in range(self.wt_levels - 1, -1, -1):
#             curr_x_ll = x_ll_in_levels.pop()
#             curr_x_h = x_h_in_levels.pop()
#             curr_shape = shapes_in_levels.pop()

#             curr_x_ll = curr_x_ll + next_x_ll

#             curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
#             next_x_ll = self.iwt_function(curr_x)

#             next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

#         x_tag = next_x_ll
#         assert len(x_ll_in_levels) == 0

#         x = self.base_scale(self.base_conv(x))
#         x = x + x_tag

#         if self.do_stride is not None:
#             x = self.do_stride(x)

#         return x


# class _ScaleModule(nn.Module):
#     def __init__(self, dims, init_scale=1.0, init_bias=0):
#         super(_ScaleModule, self).__init__()
#         self.dims = dims
#         self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
#         self.bias = None

#     def forward(self, x):
#         return torch.mul(self.weight, x)


# class DepthwiseSeparableConvWithWTConv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3):
#         super(DepthwiseSeparableConvWithWTConv2d, self).__init__()

#         # 深度卷积：使用 WTConv2d 替换 3x3 卷积
#         self.depthwise = WTConv2d(in_channels, in_channels, kernel_size=kernel_size)

#         # 逐点卷积：使用 1x1 卷积
#         self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

#     def forward(self, x):
#         x = self.depthwise(x)
#         x = self.pointwise(x)
#         return x
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        

    def forward(self, x):
        identity = x
        
        n,c,h,w = x.size()
        x_h = self.pool_h(x) # [B,C,H,1] 
        x_w = self.pool_w(x).permute(0, 1, 3, 2) # [B,C,1,W] --->  [B,C,W,1]方便等下cat时融合xy

        y = torch.cat([x_h, x_w], dim=2)#dim为2时将拼接张量里的第三个元素 即 H+W 生成的y[B,C,H+W,1]
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out
    
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


if __name__ == '__main__':
    input = torch.randn(1, 64, 128, 128)  # b c h w输入
    c2f_model = C2f(64, 128, n=2, shortcut=True)  # 创建C2f模型
    
# 前向传播测试
    # output = c2f_model(input)  # 执行前向传播
    # print(f'Output shape: {output.shape}')  # 输出形状
# #     # wtconv = DepthwiseSeparableConvWithWTConv2d(in_channels=32, out_channels=32)
#     CoordAtt = CoordAtt(inp=64, oup=64)
#     output = CoordAtt(input)
#     print(output.size())