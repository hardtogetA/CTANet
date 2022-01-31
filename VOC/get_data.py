import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch
import os
import random

all_classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
background = [0, 255]
fold_0 = [1, 2, 3, 4, 5]
fold_1 = [6, 7, 8, 9, 10]
fold_2 = [11, 12, 13, 14, 15]
fold_3 = [16, 17, 18, 19, 20]

image_path = 'E:/DataSet/Nature/VOC2012/image_resize/'
seg_label_path = 'E:/DataSet/Nature/VOC2012/label_classify/'
contour_label_path = 'E:/DataSet/Nature/VOC2012/gt_contour/'
new_label_path = 'E:/DataSet/Nature/VOC2012/new_label/'
canny_path = 'E:/DataSet/Nature/VOC2012/canny/'


def choose_fold(index):
    if index == 0:
        train_class = [x for x in all_classes if x not in fold_0]
        test_class = fold_0
    elif index == 1:
        train_class = [x for x in all_classes if x not in fold_1]
        test_class = fold_1
    elif index == 2:
        train_class = [x for x in all_classes if x not in fold_2]
        test_class = fold_2
    elif index == 3:
        train_class = [x for x in all_classes if x not in fold_3]
        test_class = fold_3
    else:
        train_class = []
        test_class = []
    return train_class, test_class


def get_oneshot_batch(fold, batch_size, train=True):
    if train:
        classes, _ = choose_fold(fold)
    else:
        _, classes = choose_fold(fold)
    support_images = np.zeros((batch_size, 3, 224, 224), dtype=np.float32)
    support_labels = np.zeros((batch_size, 1, 224, 224), dtype=np.float32)
    query_images = np.zeros((batch_size, 3, 224, 224), dtype=np.float32)
    query_canny = np.zeros((batch_size, 1, 224, 224), dtype=np.float32)
    query_contour = np.zeros((batch_size, 1, 224, 224), dtype=np.float32)
    query_new_label = np.zeros((batch_size, 1, 224, 224), dtype=np.float32)
    query_labels = np.zeros((batch_size, 1, 224, 224), dtype=np.float32)
    for i in range(batch_size):
        choose_class = random.sample(classes, 1)
        all_name = os.listdir(seg_label_path + str(choose_class[0]))
        choose_name = random.sample(all_name, 2)
        support_image = cv2.imread(image_path + choose_name[0][:-4] + '.jpg')
        if support_image is None:
            print(image_path + choose_name[0] + 'is lost')
        support_image = support_image[:, :, ::-1]
        support_image = support_image / 255.0
        std = np.array([0.229, 0.224, 0.225])
        mean = np.array([0.485, 0.456, 0.406])
        support_image = support_image - mean
        support_image = support_image / std
        support_image = np.transpose(support_image, (2, 0, 1))
        query_image = cv2.imread(image_path + choose_name[1][:-4] + '.jpg')
        if query_image is None:
            print(image_path + choose_name[1] + 'is lost')
        query_image = query_image[:, :, ::-1]  # bgr to rgb
        query_image = query_image / 255.0
        std = np.array([0.229, 0.224, 0.225])
        mean = np.array([0.485, 0.456, 0.406])
        query_image = query_image - mean
        query_image = query_image / std
        query_image = np.transpose(query_image, (2, 0, 1))
        support_label = cv2.imread(seg_label_path + str(choose_class[0]) + '/' + choose_name[0], cv2.IMREAD_GRAYSCALE)
        support_label[support_label != 0] = 1
        query_label = cv2.imread(seg_label_path + str(choose_class[0]) + '/' + choose_name[1], cv2.IMREAD_GRAYSCALE)
        query_label[query_label != 0] = 1
        canny = cv2.imread(canny_path + choose_name[1][:-4] + '.jpg', cv2.IMREAD_GRAYSCALE)
        canny[canny != 0] = 1
        contour_label = cv2.imread(contour_label_path + choose_name[1], cv2.IMREAD_GRAYSCALE)
        contour_label[contour_label != 0] = 1
        new_label = cv2.imread(new_label_path + choose_name[1], cv2.IMREAD_GRAYSCALE)
        support_images[i] = support_image
        support_labels[i][0] = support_label
        query_images[i] = query_image
        query_labels[i][0] = query_label
        query_canny[i][0] = canny
        query_contour[i][0] = contour_label
        query_new_label[i][0] = new_label
    support_images_tensor = torch.from_numpy(support_images)
    support_labels_tensor = torch.from_numpy(support_labels)
    query_images_tensor = torch.from_numpy(query_images)
    query_labels_tensor = torch.from_numpy(query_labels)
    query_canny_tensor = torch.from_numpy(query_canny)
    query_contour_tensor = torch.from_numpy(query_contour)
    query_new_label_tensor = torch.from_numpy(query_new_label).to(torch.int64)
    return support_images_tensor, support_labels_tensor, query_images_tensor, query_labels_tensor, query_contour_tensor, query_new_label_tensor, query_canny_tensor

