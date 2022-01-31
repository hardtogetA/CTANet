import torch
from torch import nn
from torchvision import models
from torch.nn import functional as F


class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch, is_mp=False, is_relu=True):
        super(conv_block, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.is_mp = is_mp
        self.is_relu = is_relu
        self.doubleconv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
        )
        self.relu = nn.ReLU(True)
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        s = self.doubleconv(x)
        z = self.relu(s)
        if self.is_mp:
            z = self.mp(z)
            return s, z
        elif self.is_relu:
            return z
        else:
            return s


class CRM(nn.Module):
    def __init__(self):
        super(CRM, self).__init__()
        self.down_block1 = conv_block(1, 32, is_mp=True)
        self.down_block2 = conv_block(32, 64, is_mp=True)
        self.down_block3 = conv_block(64, 128, is_mp=True)
        self.middle_block = conv_block(128, 256)
        self.up_block2 = conv_block(256, 128)
        self.up_block3 = conv_block(128, 64)
        self.up_block4 = conv_block(64, 32)
        self.up_sample2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.UpsamplingBilinear2d(scale_factor=2),
            # conv_block(256, 256, is_relu=False)
        )
        self.up_sample3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.UpsamplingBilinear2d(scale_factor=2),
            # conv_block(128, 128, is_relu=False)
        )
        self.up_sample4 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            nn.UpsamplingBilinear2d(scale_factor=2),
            # conv_block(64, 64, is_relu=False)
        )
        self.out_conv = nn.Conv2d(32, 2, kernel_size=1)

    def forward(self, x):
        s1, x1 = self.down_block1(x)
        s2, x2 = self.down_block2(x1)
        s3, x3 = self.down_block3(x2)
        x4 = self.middle_block(x3)
        y2 = self.up_sample2(x4)
        y2 = torch.cat([y2, s3], dim=1)
        y2 = self.up_block2(y2)
        y3 = self.up_sample3(y2)
        y3 = torch.cat([y3, s2], dim=1)
        y3 = self.up_block3(y3)
        y4 = self.up_sample4(y3)
        y4 = torch.cat([y4, s1], dim=1)
        y4 = self.up_block4(y4)
        output = self.out_conv(y4)
        out = torch.argmax(output, dim=1)
        # print('222', out.shape)
        # print('out', out.shape)
        # mask = torch.zeros_like(out, dtype=torch.int64)
        # b, _, _ = out.size()
        # for batch_index in range(0, b):
        #     # print('index', batch_index)
        #     item_out = out[batch_index]
        #     # print('before_out', torch.unique(item_out), item_out.shape)
        #     item_label = label[batch_index][0]
        #     class_num = len(torch.unique(item_label))
        #     # print('label', torch.unique(item_label), item_label.shape)
        #     item_out[item_out >= class_num] = 0
        #     # print('after_out', torch.unique(item_label), item_out.shape)
        #     mask[batch_index] = item_out
        return output, out
