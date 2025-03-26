import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional
from ODC import ODConv1d
from torchscale.architecture.config import RetNetConfig
from retnetmodel import RetNetDecoder

config1 = RetNetConfig(decoder_embed_dim=64,
                      decoder_value_embed_dim=128,
                      decoder_retention_heads=4,
                      decoder_ffn_embed_dim=128,
                      chunkwise_recurrent=True,
                      recurrent_chunk_size=10,
                      decoder_normalize_before=False,
                      # activation_fn="relu",
                      # layernorm_embedding=True,
                      decoder_layers=2,
                      no_output_layer=True,
                       dropout=0.1)
retnet = RetNetDecoder(config1)

def odconv(in_planes, out_planes, kernel, strides, reduction=0.0625, kernel_num=4):
    return ODConv1d(in_planes, out_planes, stride=strides, reduction=reduction, kernel_num=kernel_num, kernel_size=kernel, padding=0)


class Bi_lstm(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.lstm = nn.LSTM(in_channel, out_channel, bidirectional=True, batch_first=True)
        self.NiN = nn.Conv1d(int(2*in_channel), out_channel, kernel_size=1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.NiN(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)


class Dist_Block(nn.Module):
    def __init__(self, in_channel, out_channel, kernel):
        super().__init__()
        self.conv = nn.Sequential(
            odconv(in_channel, out_channel, kernel, strides=1),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(),
            nn.Dropout(0.1),
            odconv(out_channel, out_channel, kernel, strides=1),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),
            odconv(out_channel, out_channel, kernel, strides=1),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(),
            nn.Dropout(0.1),
            odconv(out_channel, out_channel, kernel, strides=1),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        y = self.conv(x)
        return y


class Dist_resBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel):
        super().__init__()
        self.conv = nn.Sequential(
            # odconv(in_channel, out_channel, kernel, strides=1),
            nn.Conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel, stride=1),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(),
            nn.Dropout(0.1),
            # odconv(in_channel, out_channel, kernel, strides=1),
            nn.Conv1d(in_channels=out_channel, out_channels=out_channel, kernel_size=kernel, stride=1),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        y = self.conv(x)
        add = x[:, :, :y.shape[2]]
        return y + add


class Dist_last_Block(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        # self.aleatoric = nn.Linear(in_channel, out_channel)
        self.MLP1 = nn.Sequential(
            nn.Linear(in_features=in_feature, out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(in_features=128, out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(in_features=128, out_features=32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.line = nn.Linear(32, out_feature)

    def forward(self, x):
        x = self.MLP1(x.reshape(x.shape[0], -1))
        x = self.line(x)
        # ale = self.aleatoric(x)
        return x


class Back_azi_model(nn.Module):
    def __init__(self):
        super().__init__()
        # distance
        # self.distance_conv = Dist_conv_Block(4, 16)
        # self.distance_conv1 = Dist_resBlock(32, 32)
        # self.distance_conv2 = Dist_resBlock(32, 16)
        # self.distance_conv3 = Dist_resBlock(16, 8)
        # self.distance_conv4 = Dist_resBlock(8, 1)

        self.distance1 = Dist_Block(3, 64, 7)
        self.distance2 = Dist_Block(64, 64, 5)

        self.res1 = Dist_resBlock(64, 64, 3)
        self.res2 = Dist_resBlock(64, 64, 3)

        self.distance_lstm = Bi_lstm(64, 64)
        self.retnet = retnet

        self.distance_last = Dist_last_Block(5632, 2)
        # self.distance_last_deep = Dist_last_Block(5632, 1)


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


    def forward(self, input):
        # dist_input = torch.cat((input, prob_p), dim=1)
        # dist_conv = self.distance_conv(dist_input)
        #
        # dist_lstm = self.distance_lstm(dist_input)
        # conv_lstm_cat = torch.cat((dist_conv, dist_lstm), dim=1)

        dist = self.distance1(input)
        dist = self.distance2(dist)

        dist = self.res1(dist)
        dist = self.res2(dist)

        dist = self.distance_lstm(dist.permute(0, 2, 1))
        dist = self.retnet(dist, None, False, False, dist)

        azi = self.distance_last(dist)

        return azi

