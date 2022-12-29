#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: LargeKernel_Block_LN.py
# Created Date: Thursday July 28th 2022
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Wednesday, 2nd November 2022 9:11:51 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2022 Shanghai Jiao Tong University
#############################################################



import torch.nn as nn
from .LayerNorm import LayerNorm2d
from .ChannelAttention import CA_layer

class LargeKernel_Block(nn.Module):
    def __init__(self, channel_num=64, bias = True, kernel_size = 19):
        super(LargeKernel_Block, self).__init__()

        norm_name       = "ln"
        act_name        = "gelu"
        pad_type        = "zero"
        hiden_ration    = 1

        if norm_name.lower() == "bn":
            norm = nn.BatchNorm2d
        
        elif norm_name.lower() == "ln":
            from .LayerNorm import LayerNorm2d
            norm = LayerNorm2d
        if act_name.lower() == "gelu":
            self.act        = nn.GELU()
        elif act_name.lower() == "leakyrelu":
            self.act        = nn.LeakyReLU(0.2)
        
        if pad_type.lower() == "zero":
            pad_ops = nn.ZeroPad2d#((padding,padding,0,0))
        elif pad_type.lower() == "reflect":
            pad_ops = nn.ReflectionPad2d#((padding,padding,0,0))
        # self.act    = nn.ReLU()
        padding     = kernel_size//2
        kernel_expand= 1
        hiden_chn   = channel_num * kernel_expand
        self.attn   = nn.Sequential(
                            LayerNorm2d(channel_num),
                            nn.Conv2d(in_channels=channel_num, out_channels=hiden_chn, kernel_size=1, bias=bias),
                            self.act,
                            # nn.Conv2d(in_channels=channel_num * kernel_expand, out_channels=channel_num * kernel_expand,
                            #         kernel_size=(kernel_size,kernel_size), padding=(padding,padding), groups=channel_num * kernel_expand, bias=bias),
                            nn.Conv2d(in_channels=hiden_chn, out_channels=hiden_chn,
                                    kernel_size=(1,kernel_size), padding=(0,padding), groups=hiden_chn, bias=bias),
                            nn.Conv2d(in_channels=hiden_chn, out_channels=hiden_chn,
                                    kernel_size=(kernel_size,1), padding=(padding,0), groups=hiden_chn, bias=bias),
                            CA_layer(hiden_chn),
                            
                            nn.Conv2d(in_channels=hiden_chn, out_channels=channel_num, kernel_size=1, bias=bias)
                        )
                        
        hidden_features = int(channel_num*hiden_ration)
        self.ln    = norm(channel_num)

        self.project_in = nn.Conv2d(channel_num, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, channel_num, kernel_size=1, bias=bias)

        
    def forward(self, x):
        res = x + self.attn(x)
        x = self.project_in(self.ln(res))
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = self.act(x1) * x2
        x = self.project_out(x)
        
        return x + res