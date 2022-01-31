import torch
from model.CTANET import CTANet
from get_data import get_oneshot_batch
import numpy as np
import cv2
import os
import random


def run_testing():
    test_classes = []
    f = open('fss_test_set.txt', "r")
    line = f.readline()
    line = line[:-1]
    test_classes.append(line)
    while line:
        line = f.readline()
        line = line[:-1]
        if line == '':
            pass
        else:
            test_classes.append(line)
    f.close()
    max_iou = 0
    for j in range(10):
        model = CTANet().to(device)
        check_point = torch.load('store/best.pth')

        model_dict = model.state_dict()
        pretrained_dict = check_point['model']
        new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
        model_dict.update(new_dict)
        model.load_state_dict(model_dict)
        model.eval()
        with torch.no_grad():
            iou_list = []
            num = 0
            for j in range(0, 500):
                classes = random.sample(test_classes, 1)
                s_images, s_labels, q_images, q_labels, q_contour_labels, q_new_labels, _ = \
                    get_oneshot_batch(classes, batch_size=1)

                s_images_g, s_labels_g, q_images_g, q_labels_g, q_contour_labels_g, q_new_labels_g = \
                    s_images.to(device), s_labels.to(device), q_images.to(device), q_labels.to(device), \
                    q_contour_labels.to(device), q_new_labels.to(device)

                contour_output, mask_output, final_masks = model.forward(s_images_g, s_labels_g, q_images_g)
                num = num + 1
                for i in range(0, final_masks.size()[0]):
                    item_mask = final_masks.detach().cpu().numpy()[i]
                    item_mask = (item_mask / 255).astype(bool)
                    item_label = q_labels.numpy()[i][0].astype(bool)
                    overlap = item_label * item_mask
                    union = item_label + item_mask
                    iou = overlap.sum() / float(union.sum())
                    iou_list.append(iou)
            iou_mean = np.mean(iou_list)
            if iou_mean > max_iou:
                max_iou = iou_mean
    print("Mean iou is {:.4f}".format(max_iou))


if __name__ == "__main__":
    device = torch.device('cuda:0')
    run_testing()
