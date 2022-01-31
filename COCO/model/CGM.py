from model import backbone
import torch
import torch.nn as nn
import torch.nn.functional as F


class residual_block(nn.Module):
    def __init__(self, ch):
        super(residual_block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.block(x)
        output = out + x

        return output


class shape_attention(nn.Module):
    def __init__(self, input1_ch, input2_ch, input1_os, input2_os):
        super(shape_attention, self).__init__()
        self.os = input2_os / input1_os
        self.res_block = residual_block(input1_ch)
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(input2_ch, input1_ch, kernel_size=1, padding=0),
            nn.BatchNorm2d(input1_ch, momentum=1, affine=True),
            nn.ReLU(inplace=True)
        )
        self.get_attention = nn.Sequential(
            nn.Conv2d(2 * input1_ch, input1_ch, kernel_size=1, padding=0),
            nn.BatchNorm2d(input1_ch, momentum=1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(input1_ch, 1, kernel_size=1, padding=0),
            nn.BatchNorm2d(1, momentum=1, affine=True),
            nn.ReLU(inplace=True),
            nn.Sigmoid(),
        )

    def forward(self, input1, input2):
        input1 = self.res_block(input1)
        _, _, h, w = input1.size()
        input2 = F.interpolate(input2, size=(h, w), mode='bilinear', align_corners=False)
        input2 = self.conv1x1(input2)
        input_feature = torch.cat([input1, input2], dim=1)
        attention_map = self.get_attention(input_feature)
        middle_feature = input_feature * attention_map
        output_feature = input_feature + middle_feature

        return output_feature


class edge_module(nn.Module):
    def __init__(self):
        super(edge_module, self).__init__()
        self.shape_attention1 = shape_attention(64, 256, 2, 4)
        self.shape_attention2 = shape_attention(128, 512, 2, 8)
        self.shape_attention3 = shape_attention(256, 1024, 2, 8)
        self.edge_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, momentum=1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, momentum=1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1, padding=0),
        )

    def forward(self, stem_f, stage1_f, stage2_f, stage3_f):
        feature1 = self.shape_attention1(stem_f, stage1_f)
        feature2 = self.shape_attention2(feature1, stage2_f)
        feature3 = self.shape_attention3(feature2, stage3_f)
        edge_mask = self.edge_head(feature3)
        edge_mask = F.interpolate(edge_mask, size=(224, 224), mode='bilinear', align_corners=False)

        return edge_mask


class CGM(nn.Module):
    def __init__(self):
        super(CGM, self).__init__()

        self.shared_backbone = backbone.backbone(pretrained=True)
        self.reduce_dim = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.edge_module = edge_module()

    def forward(self, image):
        stem_s, stage1_s, stage2_s, stage3_s, concat_feature = self.shared_backbone(image)

        contour_output = self.edge_module(stem_s, stage1_s, stage2_s, stage3_s)

        return concat_feature, contour_output

