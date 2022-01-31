from torch import optim
import logging
import torch
from model.CTANET import CTANet
from get_data import get_oneshot_batch
import numpy as np


def run_training(fold, best_checkpoint_path):
    model = CTANet().to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=MAX_EPISODE//10, gamma=0.5)
    best_metric = -0.1
    if True:
        for episode in range(START_EPISODE, MAX_EPISODE):
            model.train()
            optimizer.zero_grad()
            s_images, s_labels, q_images, q_labels, q_contour_labels, q_new_labels, _ = \
                get_oneshot_batch(fold, batch_size=8, train=True)
            s_images_g, s_labels_g, q_images_g, q_labels_g, q_contour_labels_g, q_new_labels_g = \
                s_images.to(device), s_labels.to(device), q_images.to(device), q_labels.to(device), \
                q_contour_labels.to(device), q_new_labels.to(device)
            contour_output, _, final_masks = model.forward(s_images_g, s_labels_g, q_images_g)
            loss = model.get_loss(contour_output, q_contour_labels_g, final_masks, q_labels_g)
            loss = loss.requires_grad_()
            loss.backward()
            optimizer.step()
            scheduler.step(episode)
            if (episode + 1) % 1000 == 0:
                model.eval()
                with torch.no_grad():
                    iou_list = []
                    for j in range(0, 500):
                        s_images, s_labels, q_images, q_labels, q_contour_labels, q_new_labels, _ = \
                            get_oneshot_batch(fold, batch_size=1, train=False)
                        s_images_g, s_labels_g, q_images_g, q_labels_g, q_contour_labels_g, q_new_labels_g = \
                            s_images.to(device), s_labels.to(device), q_images.to(device), q_labels.to(device), \
                            q_contour_labels.to(device), q_new_labels.to(device)
                        contour_output, _, final_masks = model.forward(s_images_g, s_labels_g, q_images_g)
                        for i in range(0, final_masks.size()[0]):
                            item_mask = final_masks.detach().cpu().numpy()[i]
                            item_mask = (item_mask / 255).astype(bool)
                            item_label = q_labels.numpy()[i][0].astype(bool)
                            overlap = item_label * item_mask
                            union = item_label + item_mask
                            iou = overlap.sum() / float(union.sum())
                            iou_list.append(iou)
                    iou_mean = np.mean(iou_list)
                is_best = False
                if iou_mean > best_metric:
                    best_episode = episode
                    best_metric = iou_mean
                    is_best = True
                else:
                    is_best = False
                checkpoint = {'model': model.state_dict()}
                if is_best:
                    torch.save(checkpoint, best_checkpoint_path)
                print('[EPISODE %05d]iou: %0.5f' % (episode,iou_mean))

    print("The best :FOLD:{}, IOU:{:.5f}".format(fold, best_metric))


if __name__ == "__main__":
    MODEL = 'VOC'
    device = torch.device('cuda:0')
    START_EPISODE = 0
    MAX_EPISODE = 100000
    LR = 0.001
    for fold in range(4):
        print("This is fold:{}, ".format(fold))
        best_checkpoint_path = 'store/fold_%s.pth' % fold
        run_training(fold=fold, best_checkpoint_path=best_checkpoint_path)
