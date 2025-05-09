import torch
from torch import Tensor


def multi_label_acc(predict: Tensor, target: Tensor):
    predict = predict.flatten()
    target = target.flatten()
    cnt = predict.shape[0]
    return torch.sum(predict == target) / cnt


def acc_topk(predict: Tensor, target: Tensor, bg, k=2):
    predict = predict.flatten()
    target = target.flatten()
    cnt = predict.shape[0]
    bg_idx = (predict == bg) + (target == bg)

    top5_correct = torch.sum(predict[bg_idx] == target[bg_idx])
    not_bg_idx = torch.logical_not(bg_idx)
    top5_correct += torch.sum(torch.abs(predict[not_bg_idx] - target[not_bg_idx]) < k)
    return top5_correct / cnt
