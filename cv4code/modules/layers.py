# SPDX-License-Identifier: Apache-2.0
import torch 
from torch import nn
from torch.autograd import Function
from einops import repeat
from vit_pytorch.cct import TransformerClassifier, Tokenizer
from typing import Iterable
from math import sqrt

class GradientReverse(Function):   
    @staticmethod 
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()
    
    @staticmethod 
    def backward(ctx, grad_output):
        lambda_ = ctx.lambda_
        return grad_output * -lambda_, None

class GradientReverseLayer(nn.Module):
    def __init__(self, lambda_):
        super(GradientReverseLayer, self).__init__()
        self.lambda_ = lambda_
    
    def forward(self, x):
        return GradientReverse().apply(x, self.lambda_)

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
    
    def forward(self, x):
        return torch.sigmoid(x) * x


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3, 
        stride=stride, padding=1, bias=False
        )

def conv3x3_1d(in_channels, out_channels, stride=1):
    return nn.Conv1d(
        in_channels, out_channels, kernel_size=3, 
        stride=stride, padding=1, bias=False
        )

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, dim=2):
        super(ResidualBlock, self).__init__()
        if dim == 2:
            self.conv1 = conv3x3(in_channels, out_channels, stride)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = conv3x3(out_channels, out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
        else:
            self.conv1 = conv3x3_1d(in_channels, out_channels, stride)
            self.bn1 = nn.BatchNorm1d(out_channels)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = conv3x3_1d(out_channels, out_channels)
            self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class SpatialPyramidPooling(nn.Module):
    def __init__(self, num_pools=[1, 4, 16], mode='max', dim=2):
        super(SpatialPyramidPooling, self).__init__()
        self.dim = dim
        if dim == 2:
            self.name = 'SpatialPyramidPooling'
        elif dim == 1:
            self.name = 'TemporalPyramidPooling'

        if mode == 'max':
            if dim == 2:
                pool_func = nn.AdaptiveMaxPool2d
            else:
                pool_func = nn.AdaptiveMaxPool1d
        elif mode == 'avg':
            if dim == 2:
                pool_func = nn.AdaptiveAvgPool2d
            else:
                pool_func = nn.AdaptiveAvgPool1d
        else:
            raise NotImplementedError(f"Unknown pooling mode '{mode}', expected 'max' or 'avg'")
        self.pools = nn.ModuleList([])
        for p in num_pools:
            side_length = sqrt(p) if dim == 2 else p
            if dim == 2 and not side_length.is_integer():
                raise ValueError(f'Bin size {p} is not a perfect square')
            self.pools.append(pool_func(int(side_length)))

    def forward(self, feature_maps):
        if self.dim == 2:
            assert feature_maps.dim() == 4, 'Expected 4D input of (N, C, H, W)'
        else:
            assert feature_maps.dim() == 3, 'Expected 3D input of (N, C, L)'
        batch_size = feature_maps.size(0)
        channels = feature_maps.size(1)
        pooled = []
        for p in self.pools:
            pooled.append(p(feature_maps).view(batch_size, channels, -1))
        return torch.cat(pooled, dim=2)

class CustomResNet(nn.Module):
    def __init__(
        self, 
        initial_in_channel, initial_out_channels, 
        blocks, 
        final_pool=['avg', [1]],
        init_fc=-1,
        init_strides=(2, 2),
        init_kernel=(7, 7),
        init_max_pool=True,
        final_fc_layers=None,
        input_dim=2):
        """
        blocks: [[n_channels, n_blocks, stride]]
        """
        super(CustomResNet, self).__init__()
        self.num_spatial_maps = sum(final_pool[1])
        self.in_channels = initial_in_channel
        self.fc = None
        conv_type = nn.Conv2d if input_dim == 2 else nn.Conv1d
        if init_fc > 0:
            self.fc = conv_type(
                initial_in_channel, init_fc, 1, stride=1, bias=False
            )
            self.in_channels = init_fc 
        self.conv = conv_type(
            self.in_channels, 
            initial_out_channels, 
            init_kernel, 
            stride=init_strides, 
            padding=(init_kernel[0]//2, init_kernel[1]//2) \
                if isinstance(init_kernel, Iterable) else init_kernel//2, 
            bias=False)
        self.in_channels = initial_out_channels
        if input_dim == 2:
            self.bn = nn.BatchNorm2d(initial_out_channels)
        else:
            self.bn = nn.BatchNorm1d(initial_out_channels)
        self.relu = nn.ReLU(inplace=True)
        if input_dim == 2:
            self.maxpool = nn.MaxPool2d(3, stride=2) if init_max_pool else None
        else:
            self.maxpool = nn.MaxPool1d(3, stride=2) if init_max_pool else None
        self._block_idx = []
        for idx, block in enumerate(blocks):
            if isinstance(block[0], str):
                setattr(self, f'block{idx}', self.make_pool(*block, dim=input_dim))
            else:
                setattr(self, f'block{idx}', self.make_layer(*block, dim=input_dim))
            self._block_idx.append(idx)
        self.global_pool = SpatialPyramidPooling(num_pools=final_pool[1], mode=final_pool[0], dim=input_dim)
        self.output_size = blocks[-1][0] * self.num_spatial_maps
        self.final_fc = None

        if final_fc_layers:
            fc_layers = []
            for n_unit in final_fc_layers:
                fc_layers.extend(
                    [nn.Linear(self.output_size, n_unit), nn.ReLU()]
                )
                self.output_size = n_unit
            self.final_fc = nn.Sequential(*fc_layers)
        
    def make_layer(self, out_channels, blocks, stride=1, dim=2):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride) if dim == 2 else \
                    conv3x3_1d(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels) if dim == 2 else \
                    nn.BatchNorm1d(out_channels))
        layers = []
        layers.append(
            ResidualBlock(self.in_channels, out_channels, stride, downsample, dim=dim))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, dim=dim))
        return nn.Sequential(*layers)

    def make_pool(self, type, kernel, stride=1, dim=2):
        pool_type = None
        if type == 'max':
            if dim == 2:
                pool_type = nn.MaxPool2d
            elif dim == 1:
                pool_type = nn.MaxPool1d
        if type == 'avg':
            if dim == 2:
                pool_type = nn.AvgPool2d
            elif dim == 1:
                pool_type = nn.AvgPool1d
        return pool_type(kernel, stride)
    
    def forward(self, x):
        if self.fc:
            x = self.fc(x)
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        if self.maxpool:
            out = self.maxpool(out)
        for idx in self._block_idx:
            out = getattr(self, f'block{idx}')(out)
        if self.global_pool:
            out = self.global_pool(out)
            out = out.view(out.size(0), -1)
        else:
            out = out.flatten(start_dim=1)
        if self.final_fc:
            out = self.final_fc(out)
        return out

class ConvTokenizer(Tokenizer):
    def __init__(self,
                n_output_channels=64,
                pad=-1, 
                **kargs):
        super(ConvTokenizer, self).__init__(
            **kargs, n_output_channels=n_output_channels)

        self.pad = pad
        self.pad_token = None
        if pad > 0:
            self.pad_token = nn.Parameter(torch.randn(1, 1, n_output_channels))

    def sequence_length(self, n_channels=3, height=224, width=224):
        if self.pad_token is None:
            return self.forward(torch.zeros((1, n_channels, height, width))).shape[1]
        else:
            original_length = self.forward(torch.zeros((1, n_channels, height, width))).shape[1]
            assert self.pad >= original_length, 'padded size shorter than the output'
            return self.pad

    def forward(self, x):
        out = self.flattener(self.conv_layers(x)).transpose(-2, -1)
        b, n, _ = out.shape
        if self.pad_token is not None and n < self.pad:
            delta_length = self.pad - n
            pad_tokens = repeat(self.pad_token, '() () d -> b n d', b = b, n = delta_length)
            out = torch.cat((out, pad_tokens), dim=1)
        return out

class ConvViT(nn.Module):
    def __init__(self,
                 img_size=224,
                 embedding_dim=768,
                 n_input_channels=3,
                 n_conv_layers=1,
                 kernel_size=7,
                 stride=2,
                 padding=3,
                 pooling_kernel_size=3,
                 pooling_stride=2,
                 pooling_padding=1,
                 token_padding=-1,
                 dropout=0.1,
                 attention_dropout=0.1,
                 seq_pool=False,
                 *args, **kwargs):
        super(ConvViT, self).__init__()

        self.tokenizer = ConvTokenizer(n_input_channels=n_input_channels,
                                   n_output_channels=embedding_dim,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   pooling_kernel_size=pooling_kernel_size,
                                   pooling_stride=pooling_stride,
                                   pooling_padding=pooling_padding,
                                   max_pool=True,
                                   activation=nn.ReLU,
                                   n_conv_layers=n_conv_layers,
                                   pad=token_padding,
                                   conv_bias=False)

        self.classifier = TransformerClassifier(
            sequence_length=self.tokenizer.sequence_length(n_channels=n_input_channels,
                                                           height=img_size,
                                                           width=img_size),
            embedding_dim=embedding_dim,
            seq_pool=seq_pool,
            dropout_rate=dropout,
            attention_dropout=attention_dropout,
            stochastic_depth=0.1,
            *args, **kwargs)

    def forward(self, x):
        x = self.tokenizer(x)
        return self.classifier(x)