import torch
from torch import nn, Tensor
from typing import Optional, Dict, Tuple, Union
from easydict import EasyDict as edict
import yaml
import argparse

from layers import norm_layers_tuple, LinearLayer, ConvLayer, GlobalPool, Identity
from misc.profiler import module_profile
from misc.init_utils import initialize_weights, initialize_fc_layer
from modules import InvertedResidual
from modules import LMSABlock as Block


def flatten_yaml_as_dict(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, edict):
            items.extend(flatten_yaml_as_dict(v, new_key))
        else:
            items.append((new_key, v))
    return items

class MoboGaze(nn.Module):
    """
    Base class for different classification models
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.loss_op = nn.L1Loss(reduce=False)
        self.l1_loss_op = nn.L1Loss()
        image_channels = config['MoboGaze']["layer0"]["img_channels"]
        out_channels = config['MoboGaze']["layer0"]["out_channels"]
        pool_type = getattr(config, "model.layer.global_pool", "mean")
        num_classes = 2
        self.model_conf_dict = dict()
        self.dilate_l4 = False
        self.dilate_l5 = False
        self.dilation = 1
        opts = edict(flatten_yaml_as_dict(config))
        self.conv_1 = ConvLayer(
            opts=opts,
            in_channels=image_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            use_norm=True,
            use_act=True,
        )
        
        self.model_conf_dict["conv1"] = {"in": image_channels, "out": out_channels}

        in_channels = out_channels
        self.layer_1, out_channels = self._make_layer(
            opts=opts, input_channel=in_channels, cfg=config['MoboGaze']["layer1"]
        )
        self.model_conf_dict["layer1"] = {"in": in_channels, "out": out_channels}

        in_channels = out_channels
        self.layer_2, out_channels = self._make_layer(
            opts=opts, input_channel=in_channels, cfg=config['MoboGaze']["layer2"]
        )
        self.model_conf_dict["layer2"] = {"in": in_channels, "out": out_channels}

        in_channels = out_channels
        self.layer_3, out_channels = self._make_layer(
            opts=opts, input_channel=in_channels, cfg=config['MoboGaze']["layer3"]
        )
        self.model_conf_dict["layer3"] = {"in": in_channels, "out": out_channels}

        in_channels = out_channels
        self.layer_4, out_channels = self._make_layer(
            opts=opts,
            input_channel=in_channels,
            cfg=config['MoboGaze']["layer4"],
            dilate=self.dilate_l4,
        )
        self.model_conf_dict["layer4"] = {"in": in_channels, "out": out_channels}

        in_channels = out_channels
        self.layer_5, out_channels = self._make_layer(
            opts=opts,
            input_channel=in_channels,
            cfg=config['MoboGaze']["layer5"],
            dilate=self.dilate_l5,
        )
        self.model_conf_dict["layer5"] = {"in": in_channels, "out": out_channels}

        self.conv_1x1_exp = Identity()
        self.model_conf_dict["exp_before_cls"] = {
            "in": out_channels,
            "out": out_channels,
        }

        self.classifier = nn.Sequential(
            GlobalPool(pool_type=pool_type, keep_dim=False),
            LinearLayer(in_features=out_channels, out_features=num_classes, bias=True),
        )

        """Initialize model weights"""
        initialize_weights(opts=opts, modules=self.modules())
    
    def _make_layer(
        self, opts, input_channel, cfg: Dict, dilate: Optional[bool] = False
    ) -> Tuple[nn.Sequential, int]:
        block_type = cfg.get("block_type", "LMSA-Vit")
        if block_type == "LMSA-Vit":
            return self._make_mit_layer(
                opts=opts, input_channel=input_channel, cfg=cfg, dilate=dilate
            )
        else:
            return self._make_mobilenet_layer(
                opts=opts, input_channel=input_channel, cfg=cfg
            )
    
    def _make_mit_layer(
        self, opts, input_channel, cfg: Dict, dilate: Optional[bool] = False
    ) -> Tuple[nn.Sequential, int]:
        prev_dilation = self.dilation
        block = []
        stride = cfg.get("stride", 1)
        if stride == 2:
            if dilate:
                self.dilation *= 2
                stride = 1

            layer = InvertedResidual(
                opts=opts,
                in_channels=input_channel,
                out_channels=cfg.get("out_channels"),
                stride=stride,
                expand_ratio=cfg.get("mv_expand_ratio", 4),
                dilation=prev_dilation,
            )

            block.append(layer)
            input_channel = cfg.get("out_channels")

        attn_unit_dim = cfg["attn_unit_dim"]
        ffn_multiplier = cfg.get("ffn_multiplier")

        dropout = getattr(opts, "model.classification.mitv2.dropout", 0.0)

        block.append(
            Block(
                opts=opts,
                in_channels=input_channel,
                attn_unit_dim=attn_unit_dim,
                ffn_multiplier=ffn_multiplier,
                n_attn_blocks=cfg.get("attn_blocks", 1),
                patch_h=cfg.get("patch_h", 2),
                patch_w=cfg.get("patch_w", 2),
                dropout=dropout,
                ffn_dropout=getattr(
                    opts, "model.classification.mitv2.ffn_dropout", 0.0
                ),
                attn_dropout=getattr(
                    opts, "model.classification.mitv2.attn_dropout", 0.0
                ),
                conv_ksize=3,
                attn_norm_layer=getattr(
                    opts, "model.classification.mitv2.attn_norm_layer", "layer_norm_2d"
                ),
                dilation=self.dilation,
            )
        )

        return nn.Sequential(*block), input_channel
        
    def _make_mobilenet_layer(self,
        opts, input_channel: int, cfg: Dict
    ) -> Tuple[nn.Sequential, int]:
        output_channels = cfg.get("out_channels")
        num_blocks = cfg.get("num_blocks", 2)
        expand_ratio = cfg.get("expand_ratio", 4)
        block = []

        for i in range(num_blocks):
            stride = cfg.get("stride", 1) if i == 0 else 1

            layer = InvertedResidual(
                opts=opts,
                in_channels=input_channel,
                out_channels=output_channels,
                stride=stride,
                expand_ratio=expand_ratio,
            )
            block.append(layer)
            input_channel = output_channels
        return nn.Sequential(*block), input_channel
    
    def extract_features(self, x: Tensor, *args, **kwargs) -> Tensor:
        x = self.conv_1(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)

        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.conv_1x1_exp(x)
        return x

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        x = self.extract_features(x)
        x = self.classifier(x)
        return x
    
    def get_weights(self, label):
        weights = torch.abs(label[:, 0]) + torch.abs(label[:, 1])
        return weights
    
    def loss(self, x_in, label):
        gaze = self.forward(x_in['face'])
        l1_loss = self.loss_op(gaze, label) 
        loss_sum = torch.sum(l1_loss, 1)
        weights = self.get_weights(label)
        loss = loss_sum * weights + loss_sum
        loss_mean = torch.mean(loss)
        
        loss2 = self.l1_loss_op(gaze, label)
        return 0.5 * loss_mean + 0.5 * loss2
    
    #def loss(self, x_in, label):
    #    gaze = self.forward(x_in['face'])
    #    loss = self.loss_op(gaze, label) 
    #    return loss

if __name__ == '__main__':
    model_config = edict(yaml.load(open('MoboGaze.yaml'), Loader=yaml.FullLoader))
    print(MoboGaze(model_config))