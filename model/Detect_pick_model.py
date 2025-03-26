import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional
from functools import partial
from typing import List
from torch import Tensor
from torchscale.architecture.config import RetNetConfig
from retnetmodel import RetNetDecoder
from Multiscale_retention import MultiScaleRetention
from Global_local_attention import AdditiveEmission
from Transformer import Encoder
from torchscale.architecture.config import EncoderConfig

"""
You can find the code of partial conv: https://github.com/JierunChen/FasterNet
and the code of RetNet: https://github.com/microsoft/torchscale

You must modify the code appropriately to fit your model.
"""
config1 = RetNetConfig(decoder_embed_dim=8,
                      decoder_value_embed_dim=32,
                      decoder_retention_heads=4,
                      decoder_ffn_embed_dim=32,
                      chunkwise_recurrent=True,
                      recurrent_chunk_size=100,
                      decoder_normalize_before=False,
                      # activation_fn="relu",
                      # layernorm_embedding=True,
                      decoder_layers=1,
                      no_output_layer=True)
retnet = RetNetDecoder(config1)

config = EncoderConfig(encoder_embed_dim=64,
                      encoder_ffn_embed_dim=128,
                      encoder_attention_heads=8,
                      encoder_layers=2,
                      no_output_layer=True)
encoder = Encoder(config)

class Embed_layer(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, pad):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel,
                      stride=1, padding=pad),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(),
        )
        self.pool = nn.MaxPool1d(2, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        return x

class decode_layer(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, pad):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel,
                      stride=1, padding=pad),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        x = self.upsample(x)
        result = self.conv1(x)
        return result





class eca_layer(nn.Module):
    def __init__(self):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, 3, 1, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)
        y = self.sigmoid(y)

        return x * y


class Res_CNN(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, pad):
        super().__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel, stride=1, padding=pad),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(in_channels=out_channel, out_channels=out_channel, kernel_size=kernel, stride=1, padding=pad),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.eca = eca_layer()

    def forward(self, x):
        res = x
        xp = self.cnn1(x)
        xp = self.eca(xp)
        xp = xp[:, :, :res.shape[2]]
        result = xp + res
        return result


class Bi_lstm(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.lstm = nn.LSTM(in_channel, out_channel, bidirectional=True, batch_first=True)
        self.NiN = nn.Conv1d(int(2*in_channel), out_channel, kernel_size=1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.NiN(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)


class lstm(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.lstm = nn.LSTM(in_channel, out_channel, batch_first=True)
        self.NiN = nn.Conv1d(out_channel, out_channel, kernel_size=1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.NiN(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)

class SRU(nn.Module):
    def __init__(self,
                 oup_channels: int,
                 group_num: int = 16,
                 gate_treshold: float = 0.5,
                 ):
        super().__init__()

        self.gn = nn.GroupNorm(num_channels=oup_channels, num_groups=group_num)
        self.gate_treshold = gate_treshold
        self.sigomid = nn.Sigmoid()

    def forward(self, x):
        gn_x = self.gn(x)
        w_gamma = self.gn.weight / sum(self.gn.weight)
        w_gamma = w_gamma.view(1, -1, 1)
        reweigts = self.sigomid(gn_x * w_gamma)
        # Gate
        w1 = torch.where(reweigts > self.gate_treshold, torch.ones_like(reweigts), reweigts)  # 大于门限值的设为1，否则保留原值
        w2 = torch.where(reweigts > self.gate_treshold, torch.zeros_like(reweigts), reweigts)  # 大于门限值的设为0，否则保留原值
        x_1 = w1 * x
        x_2 = w2 * x
        y = self.reconstruct(x_1, x_2)
        return y

    def reconstruct(self, x_1, x_2):
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)


class CRU(nn.Module):
    def __init__(self,
                 ip_channel:int,
                 op_channel: int,
                 alpha:float = 0.5,
                 squeeze_radio:int = 2 ,
                 group_size:int = 2,
                 group_kernel_size:int = 3,
                 ):
        super().__init__()
        self.up_channel     = up_channel   =   int(alpha*ip_channel)
        self.low_channel    = low_channel  =   ip_channel-up_channel
        self.squeeze1       = nn.Conv1d(up_channel,up_channel//squeeze_radio,kernel_size=1,bias=False)
        self.squeeze2       = nn.Conv1d(low_channel,low_channel//squeeze_radio,kernel_size=1,bias=False)
        #up
        self.GWC            = nn.Conv1d(up_channel//squeeze_radio, op_channel,kernel_size=group_kernel_size, stride=1,padding=group_kernel_size//2, groups = group_size)
        self.PWC1           = nn.Conv1d(up_channel//squeeze_radio, op_channel,kernel_size=1, bias=False)
        #low
        self.PWC2           = nn.Conv1d(low_channel//squeeze_radio, op_channel-low_channel//squeeze_radio,kernel_size=1, bias=False)
        self.advavg         = nn.AdaptiveAvgPool1d(1)

    def forward(self,x):
        # Split
        up,low  = torch.split(x,[self.up_channel,self.low_channel],dim=1)
        up,low  = self.squeeze1(up),self.squeeze2(low)
        # Transform
        Y1      = self.GWC(up) + self.PWC1(up)
        Y2      = torch.cat( [self.PWC2(low), low], dim= 1 )
        # Fuse
        out     = torch.cat( [Y1,Y2], dim= 1 )
        out     = torch.nn.functional.softmax( self.advavg(out), dim=1 ) * out
        out1,out2 = torch.split(out,out.size(1)//2,dim=1)
        return out1+out2


class Partial_conv3(nn.Module):

    def __init__(self, dim, n_div, kernel, pad, forward):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv1d(self.dim_conv3, self.dim_conv3, kernel_size=kernel, bias=False, padding=pad)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x: Tensor) -> Tensor:
        # only for inference
        x = x.clone()   # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])

        return x

    def forward_split_cat(self, x: Tensor) -> Tensor:
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1p = self.partial_conv3(x1)
        x1 = x1p[:, :, 0:x1.shape[2]]
        x = torch.cat((x1, x2), 1)

        return x


class MLPBlock(nn.Module):

    def __init__(self,
                 dim,
                 n_div,
                 kernel,
                 pad,
                 mlp_ratio,
                 drop_path,
                 layer_scale_init_value,
                 pconv_fw_type
                 ):

        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.drop_path = DropPath(drop_path) if drop_path > 1. else nn.Identity()
        self.n_div = n_div
        self.kernel = kernel
        self.pad = pad

        mlp_hidden_dim = int(dim * mlp_ratio)

        mlp_layer: List[nn.Module] = [
            nn.Conv1d(dim, mlp_hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(mlp_hidden_dim),
            nn.ReLU(),
            nn.Conv1d(mlp_hidden_dim, dim, kernel_size=1, bias=False)
        ]

        self.mlp = nn.Sequential(*mlp_layer)

        self.spatial_mixing = Partial_conv3(
            dim,
            n_div,
            kernel,
            pad,
            pconv_fw_type
        )

        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.forward = self.forward_layer_scale
        else:
            self.forward = self.forward

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(self.mlp(x))
        return x

    def forward_layer_scale(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(
            self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x


class embed(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, pad, model=None):
        super().__init__()
        self.embed_cnn = nn.Sequential(
            nn.Conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel,
                      stride=1, padding=pad),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        if in_channel != 3:
            self.sru = SRU(in_channel, group_num=8)
            # self.cru = CRU(in_channel, out_channel, group_kernel_size=kernel)
        else:
            self.sru = None
        self.p_conv = Partial_conv3(out_channel, n_div=4, kernel=kernel, pad=pad, forward='split_cat')
        # self.point = MLPBlock(dim=out_channel, n_div=4, kernel=kernel, pad=pad, drop_path=0, layer_scale_init_value=0,
        #                             mlp_ratio=2, pconv_fw_type='split_cat')
        if model == 'first':
            self.down_sp = nn.Conv1d(in_channels=out_channel, out_channels=out_channel, kernel_size=4, stride=4)
        else:
            self.down_sp = nn.Conv1d(in_channels=out_channel, out_channels=out_channel, kernel_size=2, stride=2)

    def forward(self, x):
        if self.sru:
            x = self.sru(x)
            x = self.embed_cnn(x)
        else:
            x = self.embed_cnn(x)
        # x = self.embed_cnn(x)
        x = self.p_conv(x)
        # x = self.point(x)
        x = self.down_sp(x)
        return x


class decode(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, pad, n, model=None):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel,
                      stride=1, padding=pad),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.p_conv = Partial_conv3(out_channel, n_div=n, kernel=kernel, pad=pad, forward='split_cat')
        # self.point = MLPBlock(dim=out_channel, n_div=n, kernel=kernel, pad=pad, drop_path=0, layer_scale_init_value=0,
        #                       mlp_ratio=2, pconv_fw_type='split_cat')
        if model == 'last':
            self.up_sp = nn.Upsample(scale_factor=4, mode='linear', align_corners=True)
        else:
            self.up_sp = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)

    def forward(self, x):
        x = self.cnn(x)
        x = self.p_conv(x)
        # x = self.point(x)
        x = self.up_sp(x)
        return x

class Detect_pick_model(nn.Module):
    def __init__(self):
        super().__init__()
        # self.ssq_embed_layer1 = ssq_embed(3, 8, 5, 2)
        # self.ssq_embed_layer2 = ssq_embed(8, 16, 3, 1)
        # self.ssq_embed_layer3 = ssq_embed(16, 32, 3, 1)
        # self.ssq_embed_layer4 = ssq_embed(32, 64, 3, 1)

        self.embed_layer1 = embed(3, 8, 11, 5)
        self.embed_layer2 = embed(8, 16, 9, 4)
        self.embed_layer3 = embed(16, 16, 7, 3)
        self.embed_layer4 = embed(16, 32, 7, 3)
        self.embed_layer5 = embed(32, 32, 5, 2)
        self.embed_layer6 = embed(32, 64, 5, 2)
        self.embed_layer7 = embed(64, 64, 3, 1)
        self.Res_CNN1 = Res_CNN(64, 64, 3, 1)
        self.Res_CNN2 = Res_CNN(64, 64, 3, 1)
        self.Res_CNN3 = Res_CNN(64, 64, 3, 1)
        self.Res_CNN4 = Res_CNN(64, 64, 2, 1)
        self.Res_CNN5 = Res_CNN(64, 64, 2, 1)
        self.Bi_lstm1 = Bi_lstm(64, 64)
        self.Bi_lstm2 = Bi_lstm(64, 64)
        self.lstm3 = lstm(64, 64)
        self.Add = AdditiveEmission(64)
        self.encoder = encoder
        # self.Retnet = retnet
        self.Retnet_p = retnet
        self.Retnet_s = retnet

        # detector part
        self.detector1 = decode_layer(64, 96, 3, 1)
        self.detector2 = decode_layer(96, 96, 5, 2)
        self.detector3 = decode_layer(96, 32, 5, 2)
        self.detector4 = decode_layer(32, 32, 7, 3)
        self.detector5 = decode_layer(32, 16, 7, 3)
        self.detector6 = decode_layer(16, 16, 9, 4)
        self.detector7 = decode_layer(16, 8, 11, 5)
        self.detector8 = nn.Conv1d(8, 1, kernel_size=11, stride=1, padding=5)
        self.detector_last = nn.Sigmoid()

        # p_picker part
        self.lstm_p = nn.LSTM(64, 64, batch_first=True)
        self.p_picker1 = decode(64, 96, 3, 1, 4)
        self.p_picker2 = decode(96, 96, 5, 2, 4)
        self.p_picker3 = decode(96, 32, 5, 2, 4)
        self.p_picker4 = decode(32, 32, 7, 3, 4)
        self.p_picker5 = decode(32, 16, 7, 3, 4)
        self.p_picker6 = decode(16, 16, 9, 4, 4)
        self.p_picker7 = decode(16, 8, 11, 5, 1)
        self.p_picker8 = nn.Conv1d(8, 1, kernel_size=11, stride=1, padding=5)
        self.p_picker_last = nn.Sigmoid()

        # s_picker part
        self.lstm_s = nn.LSTM(64, 64, batch_first=True)
        self.s_picker1 = decode(64, 96, 3, 1, 4)
        self.s_picker2 = decode(96, 96, 5, 2, 4)
        self.s_picker3 = decode(96, 32, 5, 2, 4)
        self.s_picker4 = decode(32, 32, 7, 3, 4)
        self.s_picker5 = decode(32, 16, 7, 3, 4)
        self.s_picker6 = decode(16, 16, 9, 4, 4)
        self.s_picker7 = decode(16, 8, 11, 5, 1)
        self.s_picker8 = nn.Conv1d(8, 1, kernel_size=11, stride=1, padding=5)
        self.s_picker_last = nn.Sigmoid()

        # self.liner1 = nn.Linear(78, 64)
        # self.liner2 = nn.Linear(64, 46)


    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def distance(self, p, s):
        output = torch.zeros_like(p)
        for i in range(128):
            prob = torch.zeros(6000)
            max_p = torch.argmax(p[i, 0, :])
            max_s = torch.argmax(s[i, 0, :])
            prob[max_p:max_s] = 1
            output[i, 0, :] = prob
        return output.to(p.device)


    def forward(self, input, input_ssq=None):
        # ssq_embed part
        # ssq1 = self.ssq_embed_layer1(input_ssq)
        # ssq2 = self.ssq_embed_layer2(ssq1)
        # ssq3 = self.ssq_embed_layer3(ssq2)
        # ssq = self.ssq_embed_layer4(ssq3)
        # ssq = ssq.view(ssq.size(0), ssq.size(1), -1)

        # embed part
        x1 = self.embed_layer1(input)
        x2 = self.embed_layer2(x1)
        x3 = self.embed_layer3(x2)
        x4 = self.embed_layer4(x3)
        x5 = self.embed_layer5(x4)
        x6 = self.embed_layer6(x5)
        x7 = self.embed_layer7(x6)

        # comb = torch.cat((x7, ssq), dim=2)
        # comb = self.liner1(comb)
        # comb = self.liner2(comb)
        x = self.Res_CNN1(x7)
        x = self.Res_CNN2(x)
        x = self.Res_CNN3(x)
        x = self.Res_CNN4(x)
        x = self.Res_CNN5(x)


        x = x.permute(0, 2, 1)
        x = self.Bi_lstm1(x)
        x = self.Bi_lstm2(x)
        x = self.lstm3(x)
        x = self.encoder(x, None, None, False, x, None, False, None, None)
        # x = self.Retnet(x, None, False, False, x)

        # detector part
        # d = self.Retnet(x, None, False, False, x)
        d = x.permute(0, 2, 1)
        d = self.detector1(d)
        d = torch.nn.functional.pad(d, (1, 0))
        d = self.detector2(d)
        d = torch.nn.functional.pad(d, (1, 0))
        d = self.detector3(d)
        d = torch.nn.functional.pad(d, (1, 0))
        d = self.detector4(d)
        d = self.detector5(d)
        d = self.detector6(d)
        d = self.detector7(d)
        d = self.detector8(d)
        prob_d = self.detector_last(d)

        # p_picker part
        p, _ = self.lstm_p(x)
        p = self.Add(p)
        # # p = self.add_attention_p(p, 4)
        # # p = self.Retnet_p(x, None, False, False, x)
        # p = self.Retnet_p(p, 3, False, None)
        p_attention = p.permute(0, 2, 1)
        p = self.p_picker1(p_attention)
        p = torch.nn.functional.pad(p, (1, 0))
        p = self.p_picker2(p)
        p = torch.nn.functional.pad(p, (1, 0))
        p = self.p_picker3(p)
        p = torch.nn.functional.pad(p, (1, 0))
        p = self.p_picker4(p)
        p = self.p_picker5(p)
        p = self.p_picker6(p)
        p = self.p_picker7(p)
        p = self.Retnet_p(p.permute(0, 2, 1), None, False, False, p.permute(0, 2, 1))
        p = self.p_picker8(p.permute(0, 2, 1))
        prob_p = self.p_picker_last(p)

        # s_picker part
        s, _ = self.lstm_s(x)
        s = self.Add(s)
        # # s = self.add_attention_s(s, 4)
        # # s = self.Retnet_s(x, None, False, False, x)
        # s = self.Retnet_s(s, 3, False, None)
        s_attention = s.permute(0, 2, 1)
        s = self.s_picker1(s_attention)
        s = torch.nn.functional.pad(s, (1, 0))
        s = self.s_picker2(s)
        s = torch.nn.functional.pad(s, (1, 0))
        s = self.s_picker3(s)
        s = torch.nn.functional.pad(s, (1, 0))
        s = self.s_picker4(s)
        s = self.s_picker5(s)
        s = self.s_picker6(s)
        s = self.s_picker7(s)
        s = self.Retnet_s(s.permute(0, 2, 1), None, False, False, s.permute(0, 2, 1))
        s = self.s_picker8(s.permute(0, 2, 1))
        prob_s = self.s_picker_last(s)

        return prob_d, prob_p, prob_s
