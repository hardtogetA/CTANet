import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch
import os
import random


path = 'E:/DataSet/fewshot_data/fewshot_data/'


def get_oneshot_batch(classes, batch_size):
    support_images = np.zeros((batch_size, 3, 224, 224), dtype=np.float32)
    support_labels = np.zeros((batch_size, 1, 224, 224), dtype=np.float32)
    query_images = np.zeros((batch_size, 3, 224, 224), dtype=np.float32)
    query_canny = np.zeros((batch_size, 1, 224, 224), dtype=np.float32)
    query_contour = np.zeros((batch_size, 1, 224, 224), dtype=np.float32)
    query_new_label = np.zeros((batch_size, 1, 224, 224), dtype=np.float32)
    query_labels = np.zeros((batch_size, 1, 224, 224), dtype=np.float32)

    for i in range(len(classes)):
        chosen_classes = random.sample(classes, 1)[0]
        indexs = list(range(1, 11))
        chosen_index = random.sample(indexs, 2)
        support_index = chosen_index[0]
        query_index = chosen_index[1]
        support_image = cv2.imread(path + '/%s/%s.jpg' % (chosen_classes, support_index))
        if support_image is None:
            print(path + '/%s/%s.jpg' % (chosen_classes, support_index) + 'is lost')
        support_image = support_image[:, :, ::-1]
        support_image = support_image / 255.0
        std = np.array([0.229, 0.224, 0.225])
        mean = np.array([0.485, 0.456, 0.406])
        support_image = support_image - mean
        support_image = support_image / std
        support_image = np.transpose(support_image, (2, 0, 1))
        query_image = cv2.imread(path + '/%s/%s.jpg' % (chosen_classes, query_index))
        if query_image is None:
            print(path + '/%s/%s.jpg' % (chosen_classes, query_index) + 'is lost')
        query_image = query_image[:, :, ::-1]
        query_image = query_image / 255.0
        std = np.array([0.229, 0.224, 0.225])
        mean = np.array([0.485, 0.456, 0.406])
        query_image = query_image - mean
        query_image = query_image / std
        query_image = np.transpose(query_image, (2, 0, 1))
        support_label = cv2.imread(path + '/%s/%s.png' % (chosen_classes, support_index), cv2.IMREAD_GRAYSCALE)
        support_label[support_label != 0] = 1
        query_label = cv2.imread(path + '/%s/%s.png' % (chosen_classes, query_index), cv2.IMREAD_GRAYSCALE)
        query_label[query_label != 0] = 1
        canny = cv2.imread(path + '/%s/%s_canny.png' % (chosen_classes, query_index), cv2.IMREAD_GRAYSCALE)
        canny[canny != 0] = 1
        contour_label = cv2.imread(path + '/%s/%s_edge.png' % (chosen_classes, query_index), cv2.IMREAD_GRAYSCALE)
        contour_label[contour_label != 0] = 1
        new_label = cv2.imread(path + '/%s/%s.png' % (chosen_classes, query_index), cv2.IMREAD_GRAYSCALE)
        new_label[new_label != 0] = 1
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


