#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)
# Modified from 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker)

from collections import OrderedDict

import torch
import torch.nn.functional as F
import torch.nn as nn

from components import (
    DenseLayer,
    StatsPool,
    TDNNLayer,
    CAMDenseTDNNBlock,
    TransitLayer,
    get_nonlinear,
    FCM,
)


class CAMPPlus(torch.nn.Module):
    def __init__(
            self,
            feat_dim=80,
            embedding_size=192,
            growth_rate=32,
            bn_size=4,
            init_channels=128,
            config_str="batchnorm-relu",
            memory_efficient=True,
            output_level="segment",
    ):
        super().__init__()

        self.head = FCM(feat_dim=feat_dim)
        channels = self.head.out_channels
        self.output_level = output_level

        self.xvector = torch.nn.Sequential(
            OrderedDict(
                [
                    (
                        "tdnn",
                        TDNNLayer(
                            channels,
                            init_channels,
                            5,
                            stride=2,
                            dilation=1,
                            padding=-1,
                            config_str=config_str,
                        ),
                    ),
                ]
            )
        )
        channels = init_channels
        for i, (num_layers, kernel_size, dilation) in enumerate(
                zip((12, 24, 16), (3, 3, 3), (1, 2, 2))
        ):
            block = CAMDenseTDNNBlock(
                num_layers=num_layers,
                in_channels=channels,
                out_channels=growth_rate,
                bn_channels=bn_size * growth_rate,
                kernel_size=kernel_size,
                dilation=dilation,
                config_str=config_str,
                memory_efficient=memory_efficient,
            )
            self.xvector.add_module("block%d" % (i + 1), block)
            channels = channels + num_layers * growth_rate
            self.xvector.add_module(
                "transit%d" % (i + 1),
                TransitLayer(channels, channels // 2, bias=False, config_str=config_str),
            )
            channels //= 2

        self.xvector.add_module("out_nonlinear", get_nonlinear(config_str, channels))

        if self.output_level == "segment":
            self.xvector.add_module("stats", StatsPool())
            self.xvector.add_module(
                "dense", DenseLayer(channels * 2, embedding_size, config_str="batchnorm_")
            )
        else:
            assert (
                    self.output_level == "frame"
            ), "`output_level` should be set to 'segment' or 'frame'. "

        for m in self.modules():
            if isinstance(m, (torch.nn.Conv1d, torch.nn.Linear)):
                torch.nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B,T,F) => (B,F,T)
        x = self.head(x)
        x = self.xvector(x)
        if self.output_level == "frame":
            x = x.transpose(1, 2)
        return x


class CosineClassifier(nn.Module):
    def __init__(
            self,
            input_dim,
            num_blocks=0,
            inter_dim=512,
            out_neurons=1000,
    ):

        super().__init__()
        self.blocks = nn.ModuleList()

        for index in range(num_blocks):
            self.blocks.append(
                DenseLayer(input_dim, inter_dim, config_str='batchnorm')
            )
            input_dim = inter_dim

        self.weight = nn.Parameter(
            torch.FloatTensor(out_neurons, input_dim)
        )
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        # x: [B, dim]
        for layer in self.blocks:
            x = layer(x)

        # normalized
        x = F.linear(F.normalize(x), F.normalize(self.weight))
        return x


class CAMPPlusClassifier(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.encoder = CAMPPlus()
        self.classifier = CosineClassifier(192, out_neurons=num_class)

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x
