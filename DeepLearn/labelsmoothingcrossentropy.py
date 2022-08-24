import torch
from torch import nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing, classes):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.classes = classes
        self.KLDivLoss = nn.KLDivLoss(reduction='batchmean')

    def _get_label_smooth(self, labels):
        device = labels.device
        true_labels = F.one_hot(labels, self.classes).detach().cpu()  # 变换为one-hot编码
        assert 0 <= self.smoothing < 1
        confidence = 1.0 - self.smoothing
        with torch.no_grad():
            true_dist = torch.empty(size=(true_labels.size(0), self.classes), device=true_labels.device)
            # # gt下降smoothing，smoothing平分给其他非gt类，所以这里计算出每个非gt类的值后全部填充，而gt值后面再修改
            true_dist.fill_(self.smoothing / (self.classes - 1))
            _, index = torch.max(true_labels, 1)  # 获取最大值pos

            # input.scatter_(dim, index, src)：将src中数据根据index中的索引按照dim的方向填进input
            true_dist.scatter_(1, torch.LongTensor(index.unsqueeze(1)), confidence)
        return true_dist.to(device)

    def forward(self, outputs, labels):
        """
        参考：https://blog.csdn.net/FY_2018/article/details/119716777
        :param outputs: 模型的直接输出
        :param labels: gt值
        :return:
        """
        label_smooth = self._get_label_smooth(labels)   # 获取平滑后的label
        return self.KLDivLoss(F.log_softmax(outputs, dim=1), label_smooth)


if __name__ == "__main__":
    # labels.shape =(bs,) int64
    # outputs.shape=(bs,classes) float32
    labels = torch.tensor([0, 1], dtype=torch.int64, device="cpu")
    outputs = torch.tensor([[0.6, 0.1, 0.1, 0.1, 0.1], [0.1, 0.6, 0.1, 0.1, 0.1]], dtype=torch.float32, device="cpu")

    loss_fn=LabelSmoothingCrossEntropy(smoothing=0.1,classes=5)
    loss = loss_fn(outputs, labels)  # 计算loss
    print(loss)
