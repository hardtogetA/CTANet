from model.backbone import backbone
from model.CGM import CGM
from model.CRM import CRM
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from others.loss.loss import *


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
        start = x
        x = self.block(start)
        output = start + x

        return output


def cal_similarity(s_prototype, q_prototype):
    similarity = torch.cosine_similarity(s_prototype, q_prototype, dim=0).unsqueeze(dim=1).unsqueeze(dim=2)
    if similarity.item() >= 0:
        return True
    else:
        return False


class Contour_Net(nn.Module):
    def __init__(self):
        super(Contour_Net, self).__init__()

        self.cgm = CGM()
        self.crm = CRM()
        self.contour_threshold = 0.90

    def forward(self, query_image):
        concat_feature, cgm_output = self.cgm(query_image)
        primary_contour = F.sigmoid(cgm_output)
        primary_contour[primary_contour <= self.contour_threshold] = 0
        primary_contour[primary_contour > self.contour_threshold] = 1
        contour_output = primary_contour * 255
        crm_output, query_mask = self.crm(primary_contour)
        return concat_feature, contour_output, query_mask


class CTANet(nn.Module):
    def __init__(self):
        super(CTANet, self).__init__()
        self.contour_predict_net = Contour_Net()
        self.backbone_stage4 = nn.Sequential(
            nn.Conv2d(1536, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=1, affine=True),
            nn.ReLU(),
        )
        self.loss1 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([100]))
        self.loss2 = loss_c1()
        self.loss3 = nn.CrossEntropyLoss()
        self.loss4 = nn.BCELoss()
        self.device = torch.device('cuda:0')
        self.contour_threshold = 0.95

    def forward(self, support_image, support_label, query_image):
        concat_s, _, _ = self.contour_predict_net(support_image)
        s_feature = self.backbone_stage4(concat_s)
        concat_q, contour_output, query_mask = self.contour_predict_net(query_image)
        q_feature = self.backbone_stage4(concat_q)
        final_q_mask = torch.zeros_like(query_mask)
        small_s_label = F.interpolate(support_label, size=(28, 28), mode='nearest')
        query_mask = torch.unsqueeze(query_mask, dim=1)
        small_q_mask = F.interpolate(query_mask.float(), size=(28, 28), mode='nearest')
        s_prototype = torch.sum(torch.sum(s_feature * small_s_label, dim=3), dim=2) / torch.sum(small_s_label)
        s_prototype = s_prototype.unsqueeze(dim=2).unsqueeze(dim=3)
        b, _, _, _ = small_s_label.size()
        for batch_index in range(0, b):
            item_s_prototype = s_prototype[batch_index]
            item_q_feature = q_feature[batch_index]
            item_small_q_mask = small_q_mask[batch_index]
            item_big_q_mask = query_mask[batch_index].clone().detach()
            item_q_mask_unique = torch.unique(item_small_q_mask)
            if item_q_mask_unique[0] == 0:
                item_q_mask_unique = item_q_mask_unique[1:]
                if len(item_q_mask_unique) == 1:
                    item_big_q_mask[item_big_q_mask == item_q_mask_unique[0]] = 255
                elif len(item_q_mask_unique) == 0:
                    item_big_q_mask[item_big_q_mask != 0] = 0
                else:
                    for value in item_q_mask_unique:
                        if value != 255:
                            item_item_q_mask = item_small_q_mask.clone()
                            item_item_q_mask[item_item_q_mask != value] = 0
                            q_prototype = torch.sum(torch.sum(item_q_feature * item_item_q_mask, dim=2),
                                                    dim=1) / torch.sum(item_item_q_mask)
                            q_prototype = q_prototype.unsqueeze(dim=1).unsqueeze(dim=2)
                            is_same_class = cal_similarity(item_s_prototype, q_prototype)
                            if is_same_class:
                                item_big_q_mask[item_big_q_mask == value] = 255
                            else:
                                item_big_q_mask[item_big_q_mask == value] = 0
            item_big_q_mask[item_big_q_mask != 255] = 0
            final_q_mask[batch_index] = item_big_q_mask
        return contour_output, query_mask, final_q_mask

    def get_loss(self, contour_output, contour_gt, mask, seg_label):
        contour_output = (contour_output / 255).type(torch.int64)
        contour_output = torch.nn.functional.one_hot(contour_output, 2).type(torch.float32)
        contour_output = torch.squeeze(contour_output, dim=1).permute(0, 3, 1, 2)
        mask = mask / 255.0
        return self.loss4(mask, seg_label) + self.loss2(contour_output, contour_gt)


