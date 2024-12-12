# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Convolution modules."""

import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

__all__ = (
    "Conv",
    "Conv2",
    "LightConv",
    "DWConv",
    "DWConvTranspose2d",
    "ConvTranspose",
    "Focus",
    "GhostConv",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "Concat",
    "RepConv",
    "AdaptiveRotatedConv2d"
    "RotateConv"
)


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


class Conv2(Conv):
    """Simplified RepConv module with Conv fusing."""

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """Apply fused convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0] : i[0] + 1, i[1] : i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__("cv2")
        self.forward = self.forward_fuse


class LightConv(nn.Module):
    """
    Light convolution with args(ch_in, ch_out, kernel).

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """Apply 2 convolutions to input tensor."""
        return self.conv2(self.conv1(x))


class DWConv(Conv):
    """Depth-wise convolution."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        """Initialize Depth-wise convolution with given parameters."""
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):  # ch_in, ch_out, kernel, stride, padding, padding_out
        """Initialize DWConvTranspose2d class with given parameters."""
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """Convolution transpose 2d layer."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """Initialize ConvTranspose2d layer with batch normalization and activation function."""
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Applies transposed convolutions, batch normalization and activation to input."""
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """Applies activation and convolution transpose operation to input."""
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """Focus wh information into c-space."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initializes Focus object with user defined channel, convolution, padding, group and activation values."""
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """
        Applies convolution to concatenated tensor and returns the output.

        Input shape is (b,c,w,h) and output shape is (b,4c,w/2,h/2).
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """Ghost Convolution https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initializes Ghost Convolution module with primary and cheap operations for efficient feature learning."""
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """Forward propagation through a Ghost Bottleneck layer with skip connection."""
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class RepConv(nn.Module):
    """
    RepConv is a basic rep-style block, including training and deploy status.

    This module is used in RT-DETR.
    Based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """Initializes Light Convolution layer with inputs, outputs & optional activation function."""
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward process."""
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward process."""
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """Returns equivalent kernel and bias by adding 3x3 kernel, 1x1 kernel and identity kernel with their biases."""
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        """Pads a 1x1 tensor to a 3x3 tensor."""
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """Generates appropriate kernels and biases for convolution by fusing branches of the neural network."""
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Combines two convolution layers into a single layer and removes unused attributes from the class."""
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")


class ChannelAttention(nn.Module):
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""

    def __init__(self, channels: int) -> None:
        """Initializes the class and sets the basic configurations and instance variables required."""
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies forward pass using activation on convolutions of the input, optionally using batch normalization."""
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """Spatial-attention module."""

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given input channel (c1) and kernel size."""
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Applies the forward pass through C1 module."""
        return self.spatial_attention(self.channel_attention(x))


class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        return torch.cat(x, self.d)

####seekyou-12-10-2024
#adaptive_rotate_conv ./adaptive_rotate_conv
import einops
import math
import warnings

def _trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    # Values are generated by using a truncated uniform distribution and
    # then using the inverse CDF for the normal distribution.
    # Get upper and lower cdf values
    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)

    # Uniformly fill tensor with values from [l, u], then translate to
    # [2l-1, 2u-1].
    tensor.uniform_(2 * l - 1, 2 * u - 1)

    # Use inverse cdf transform for normal distribution to get truncated
    # standard normal
    tensor.erfinv_()

    # Transform to proper mean, std
    tensor.mul_(std * math.sqrt(2.))
    tensor.add_(mean)

    # Clamp to ensure it's in the proper range
    tensor.clamp_(min=a, max=b)
    return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    NOTE: this impl is similar to the PyTorch trunc_normal_, the bounds [a, b] are
    applied while sampling the normal with mean/std applied, therefore a, b args
    should be adjusted to match the range of mean, std args.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    with torch.no_grad():
        return _trunc_normal_(tensor, mean, std, a, b)


class LayerNormProxy(nn.Module):
    # copy from https://github.com/LeapLabTHU/DAT/blob/main/models/dat_blocks.py
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')
    

class RountingFunction(nn.Module):

    def __init__(self, in_channels, kernel_number, dropout_rate=0.2, proportion=40.0):
        super().__init__()
        self.kernel_number = kernel_number
        self.dwc = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,
                             groups=in_channels, bias=False)
        self.norm = LayerNormProxy(in_channels)
        self.relu = nn.ReLU(inplace=True)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc_alpha = nn.Linear(in_channels, kernel_number, bias=True)

        self.dropout2= nn.Dropout(dropout_rate)
        self.fc_theta = nn.Linear(in_channels, kernel_number, bias=False)

        self.act_func = nn.Softsign()
        self.proportion = proportion / 180.0 * math.pi
        
        # init weights
        trunc_normal_(self.dwc.weight, std=.02)
        trunc_normal_(self.fc_alpha.weight, std=.02)
        trunc_normal_(self.fc_theta.weight, std=.02)

    def forward(self, x):

        x = self.dwc(x)
        x = self.norm(x)
        x = self.relu(x)

        x = self.avg_pool(x).squeeze(dim=-1).squeeze(dim=-1)  # avg_x.shape = [batch_size, Cin]

        alphas = self.dropout1(x)
        alphas = self.fc_alpha(alphas)
        alphas = torch.sigmoid(alphas)

        angles = self.dropout2(x)
        angles = self.fc_theta(angles)
        angles = self.act_func(angles)#softsign范围[-1，1]后面乘以40说明角度映射为[-40,40]不过不知道为啥设置为40
        angles = angles * self.proportion

        return alphas, angles

    def extra_repr(self):
        s = (f'kernel_number={self.kernel_number}')
        return s.format(**self.__dict__)


def _get_rotation_matrix(thetas):
    bs, g = thetas.shape
    device = thetas.device
    thetas = thetas.reshape(-1)  # [bs, n] --> [bs x n]
    
    x = torch.cos(thetas)
    y = torch.sin(thetas)
    x = x.unsqueeze(0).unsqueeze(0)  # shape = [1, 1, bs * g]
    y = y.unsqueeze(0).unsqueeze(0)
    a = x - y
    b = x * y
    c = x + y

    rot_mat_positive = torch.cat((
        torch.cat((a, 1-a, torch.zeros(1, 7, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 1, bs*g, device=device), x-b, b, torch.zeros(1, 1, bs*g, device=device), 1-c+b, y-b, torch.zeros(1, 3, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 2, bs*g, device=device), a, torch.zeros(1, 2, bs*g, device=device), 1-a, torch.zeros(1, 3, bs*g, device=device)), dim=1),
        torch.cat((b, y-b, torch.zeros(1,1 , bs*g, device=device), x-b, 1-c+b, torch.zeros(1, 4, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 4, bs*g, device=device), torch.ones(1, 1, bs*g, device=device), torch.zeros(1, 4, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 4, bs*g, device=device), 1-c+b, x-b, torch.zeros(1, 1, bs*g, device=device), y-b, b), dim=1),
        torch.cat((torch.zeros(1, 3, bs*g, device=device), 1-a, torch.zeros(1, 2, bs*g, device=device), a, torch.zeros(1, 2, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 3, bs*g, device=device), y-b, 1-c+b, torch.zeros(1, 1, bs*g, device=device), b, x-b, torch.zeros(1, 1, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 7, bs*g, device=device), 1-a, a), dim=1)
    ), dim=0)  # shape = [k^2, k^2, bs*g]

    rot_mat_negative = torch.cat((
        torch.cat((c, torch.zeros(1, 2, bs*g, device=device), 1-c, torch.zeros(1, 5, bs*g, device=device)), dim=1),
        torch.cat((-b, x+b, torch.zeros(1, 1, bs*g, device=device), b-y, 1-a-b, torch.zeros(1, 4, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 1, bs*g, device=device), 1-c, c, torch.zeros(1, 6, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 3, bs*g, device=device), x+b, 1-a-b, torch.zeros(1, 1, bs*g, device=device), -b, b-y, torch.zeros(1, 1, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 4, bs*g, device=device), torch.ones(1, 1, bs*g, device=device), torch.zeros(1, 4, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 1, bs*g, device=device), b-y, -b, torch.zeros(1, 1, bs*g, device=device), 1-a-b, x+b, torch.zeros(1, 3, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 6, bs*g, device=device), c, 1-c, torch.zeros(1, 1, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 4, bs*g, device=device), 1-a-b, b-y, torch.zeros(1, 1, bs*g, device=device), x+b, -b), dim=1),
        torch.cat((torch.zeros(1, 5, bs*g, device=device), 1-c, torch.zeros(1, 2, bs*g, device=device), c), dim=1)
    ), dim=0)  # shape = [k^2, k^2, bs*g]

    mask = (thetas >= 0).unsqueeze(0).unsqueeze(0)
    mask = mask.float()                                                   # shape = [1, 1, bs*g]
    rot_mat = mask * rot_mat_positive + (1 - mask) * rot_mat_negative     # shape = [k*k, k*k, bs*g]
    rot_mat = rot_mat.permute(2, 0, 1)                                    # shape = [bs*g, k*k, k*k]
    rot_mat = rot_mat.reshape(bs, g, rot_mat.shape[1], rot_mat.shape[2])  # shape = [bs, g, k*k, k*k]
    return rot_mat


def batch_rotate_multiweight(weights, lambdas, thetas):
    """
    Let
        batch_size = b
        kernel_number = n
        kernel_size = 3
    Args:
        weights: tensor, shape = [kernel_number, Cout, Cin, k, k]
        thetas: tensor of thetas,  shape = [batch_size, kernel_number]
    Return:
        weights_out: tensor, shape = [batch_size x Cout, Cin // groups, k, k]
    """
    assert(thetas.shape == lambdas.shape)
    assert(lambdas.shape[1] == weights.shape[0])

    b = thetas.shape[0]
    n = thetas.shape[1]
    k = weights.shape[-1]
    _, Cout, Cin, _, _ = weights.shape

    # Stage 1:
    # input: thetas: [b, n]
    #        lambdas: [b, n]
    # output: rotation_matrix: [b, n, 9, 9] (with gate) --> [b*9, n*9]

    #       Sub_Stage 1.1:
    #       input: [b, n] kernel
    #       output: [b, n, 9, 9] rotation matrix
    rotation_matrix = _get_rotation_matrix(thetas)

    #       Sub_Stage 1.2:
    #       input: [b, n, 9, 9] rotation matrix
    #              [b, n] lambdas
    #          --> [b, n, 1, 1] lambdas
    #          --> [b, n, 1, 1] lambdas dot [b, n, 9, 9] rotation matrix
    #          --> [b, n, 9, 9] rotation matrix with gate (done)
    #       output: [b, n, 9, 9] rotation matrix with gate
    lambdas = lambdas.unsqueeze(2).unsqueeze(3)
    rotation_matrix = torch.mul(rotation_matrix, lambdas)

    #       Sub_Stage 1.3: Reshape
    #       input: [b, n, 9, 9] rotation matrix with gate
    #       output: [b*9, n*9] rotation matrix with gate
    rotation_matrix = rotation_matrix.permute(0, 2, 1, 3)
    rotation_matrix = rotation_matrix.reshape(b*9, n*9)

    # Stage 2: Reshape 
    # input: weights: [n, Cout, Cin, 3, 3]
    #             --> [n, 3, 3, Cout, Cin]
    #             --> [n*9, Cout*Cin] done
    # output: weights: [n*9, Cout*Cin]
    weights = weights.permute(0, 3, 4, 1, 2)
    weights = weights.contiguous().view(n*9, Cout*Cin)


    # Stage 3: torch.mm
    # [b*9, n*9] x [n*9, Cout*Cin]
    # --> [b*9, Cout*Cin]
    weights = torch.mm(rotation_matrix, weights)

    # Stage 4: Reshape Back
    # input: [b*9, Cout*Cin]
    #    --> [b, 3, 3, Cout, Cin]
    #    --> [b, Cout, Cin, 3, 3]
    #    --> [b * Cout, Cin, 3, 3] done
    # output: [b * Cout, Cin, 3, 3]
    weights = weights.contiguous().view(b, 3, 3, Cout, Cin)
    weights = weights.permute(0, 3, 4, 1, 2)
    weights = weights.reshape(b * Cout, Cin, 3, 3)

    return weights

"""Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
class AdaptiveRotatedConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=1, dilation=1, groups=1, bias=False,
                 kernel_number=1, rounting_func=None, rotate_func=batch_rotate_multiweight):
        super().__init__()
        self.kernel_number = kernel_number
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        self.rounting_func = rounting_func
        self.rotate_func = rotate_func

        self.weight = nn.Parameter(
            torch.Tensor(
                kernel_number, 
                out_channels,
                in_channels // groups,
                kernel_size,
                kernel_size,
            )
        )
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        # get alphas, angles
        # # [bs, Cin, h, w] --> [bs, n_theta], [bs, n_theta]
        alphas, angles = self.rounting_func(x)

        # rotate weight
        # # [Cout, Cin, k, k] --> [bs * Cout, Cin, k, k]
        rotated_weight = self.rotate_func(self.weight, alphas, angles)

        # reshape images
        bs, Cin, h, w = x.shape
        x = x.reshape(1, bs * Cin, h, w)  # [1, bs * Cin, h, w]
        
        # adaptive conv over images using group conv
        out = F.conv2d(input=x, weight=rotated_weight, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=(self.groups * bs))
        
        # reshape back
        out = out.reshape(bs, self.out_channels, *out.shape[2:])
        return out

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_number={kernel_number}'
             ', kernel_size={kernel_size}, stride={stride}, bias={bias}')
             
        if self.padding != (0,) * len([self.padding]):
            s += ', padding={padding}'
        if self.dilation != (1,) * len([self.dilation]):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        return s.format(**self.__dict__)
    
####seekyou-12-10-2024
class RotateConv(nn.Module):
    """Standard RotateConv with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, kernel_number, activation).基于Conv改的"""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, kernel_number=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.kernel_number = kernel_number
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.conv2 = AdaptiveRotatedConv2d(in_channels=c1,
                out_channels=c2,
                kernel_size=k, 
                stride=s,
                groups=g,
                dilation=d,
                rounting_func=RountingFunction(
                    in_channels=c1,
                    kernel_number=kernel_number,
                ),
                kernel_number=kernel_number,
                )
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.compress = nn.Conv2d(2 * c2, c2, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        normal_conv_output = self.conv(x)  # [batch_size, c2, H, W]
        # Apply rotated convolution
        rotated_conv_output = self.conv2(x)  # [batch_size, c2, H, W]
        # Concatenate the results along the channel dimension (dim=1)
        concat_output = torch.cat((normal_conv_output, rotated_conv_output), dim=1)
        compressed_output = self.compress(concat_output)  # [batch_size, c2, H, W]
        # Apply batch normalization and activation
        out = self.act(self.bn(compressed_output))
        return out #orgin':conv

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))
####seekyou-12-10-2024
