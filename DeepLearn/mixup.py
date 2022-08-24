import torch
import numpy as np
from torch.autograd import Variable

# https://github.com/facebookresearch/mixup-cifar10/blob/eaff31ab397a90fbc0a4aac71fb5311144b3608b/train.py#L137
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        # 将[0,n)随机打乱后获得的数字序列，函数名是random permutation缩写
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    # x是按顺序的batch，而index是x打乱顺序后的，然后两两mixup，如下所示，即batch内mixup
    #   0  +  2
    #   1  +  0
    #   2  +  3
    #   3  +  1
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """

    :param criterion: 损失计算函数
    :param pred: groundtrue targets
    :param y_a: 顺序batch的target
    :param y_b: 乱序batch的target
    :param lam:
    :return:
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


if __name__ == "__main__":
    device = "cuda:0"
    inputs = torch.ones([8, 3, 224, 224], dtype=torch.float32, device=device)
    targets = torch.tensor([5, 2, 1, 4, 3, 8, 5, 1], dtype=torch.int64, device=device)
    inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, 0.2, use_cuda=True)
    inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))
    outputs = net(inputs)
    loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
