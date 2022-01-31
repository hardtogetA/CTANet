import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt


def softmax_helper(x):
    rpt = [1 for _ in range(len(x.size()))]
    rpt[1] = x.size(1)
    x_max = x.max(1, keepdim=True)[0].repeat(*rpt)
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum(1, keepdim=True).repeat(*rpt)


def compute_edts_forPenalizedLoss(GT):
    res = np.zeros(GT.shape)
    for i in range(GT.shape[0]):
        posmask = GT[i]
        negmask = ~posmask
        pos_edt = distance_transform_edt(posmask)
        pos_edt = (np.max(pos_edt) - pos_edt) * posmask
        neg_edt = distance_transform_edt(negmask)
        neg_edt = (np.max(neg_edt) - neg_edt) * negmask

        res[i] = pos_edt / np.max(pos_edt) + neg_edt / np.max(neg_edt)
    return res


def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp


def get_tp_fp_fn(net_output, gt, axes=None, mask=None, square=False):
    def compute_edts_forhdloss(segmentation):
        res = np.zeros(segmentation.shape)
        for i in range(segmentation.shape[0]):
            posmask = segmentation[i]
            negmask = ~posmask
            res[i] = distance_transform_edt(posmask) + distance_transform_edt(negmask)
        return res


class loss_s(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.,
                 square=False):
        super(loss_s, self).__init__()

        self.square = square
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.r = 0.1  # weight parameter in SS paper

    def forward(self, net_output, gt, loss_mask=None):
        shp_x = net_output.shape
        shp_y = gt.shape
        # class_num = shp_x[1]

        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                gt = gt.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = gt
            else:
                gt = gt.long()
                y_onehot = torch.zeros(shp_x)
                if net_output.device.type == "cuda":
                    y_onehot = y_onehot.cuda(net_output.device.index)
                y_onehot.scatter_(1, gt, 1)

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            softmax_output = self.apply_nonlin(net_output)

        # no object value
        bg_onehot = 1 - y_onehot
        squared_error = (y_onehot - softmax_output) ** 2
        specificity_part = sum_tensor(squared_error * y_onehot, axes) / (sum_tensor(y_onehot, axes) + self.smooth)
        sensitivity_part = sum_tensor(squared_error * bg_onehot, axes) / (sum_tensor(bg_onehot, axes) + self.smooth)

        ss = self.r * specificity_part + (1 - self.r) * sensitivity_part

        if not self.do_bg:
            if self.batch_dice:
                ss = ss[1:]
            else:
                ss = ss[:, 1:]
        ss = ss.mean()

        return ss


class loss_c1(nn.Module):
    def __init__(self, smooth=1e-5):
        super(loss_c1, self).__init__()
        self.smooth = smooth

    def forward(self, net_output, gt):
        net_output = softmax_helper(net_output)
        # one hot code for gt
        with torch.no_grad():
            if len(net_output.shape) != len(gt.shape):
                gt = gt.view((gt.shape[0], 1, *gt.shape[1:]))

            if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = gt
            else:
                gt = gt.long()
                y_onehot = torch.zeros(net_output.shape)
                print(y_onehot.shape)
                if net_output.device.type == "cuda":
                    y_onehot = y_onehot.cuda(net_output.device.index)
                y_onehot.scatter_(1, gt, 1)

        gt_temp = gt[:, 0, ...].type(torch.float32)
        with torch.no_grad():
            dist = compute_edts_forPenalizedLoss(gt_temp.cpu().numpy() > 0.5) + 1.0
        # print('dist.shape: ', dist.shape)
        dist = torch.from_numpy(dist)

        if dist.device != net_output.device:
            dist = dist.to(net_output.device).type(torch.float32)

        tp = net_output * y_onehot
        tp = torch.sum(tp[:, 1, ...] * dist, (1, 2))

        dc = (2 * tp + self.smooth) / (torch.sum(net_output[:, 1, ...], (1, 2)) + torch.sum(y_onehot[:, 1, ...],
                                                                                               (1, 2)) + self.smooth)

        dc = dc.mean()

        return dc


class loss_t(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.,
                 square=False):
        super(loss_t, self).__init__()

        self.square = square
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.alpha = 0.3
        self.beta = 0.7

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn = get_tp_fp_fn(x, y, axes, loss_mask, self.square)

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        if not self.do_bg:
            if self.batch_dice:
                tversky = tversky[1:]
            else:
                tversky = tversky[:, 1:]
        tversky = tversky.mean()

        return -tversky


class loss_c(nn.Module):
    def __init__(self, smooth=1e-5):
        super(loss_c, self).__init__()
        self.smooth = smooth

    def forward(self, mask, label):
        B = mask.shape[0]
        A = 0.0
        for i in range(0, B):
            X = torch.sum(label[i, :, :, :] * mask[i, :, :, :])
            Y = torch.sum(label[i, :, :, :]) + torch.sum(mask[i, :, :, :]) - X + self.smooth
            Z = X / Y

            A = A + Z.item()

        return A / B



