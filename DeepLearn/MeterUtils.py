class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
       
        # batch_time=AverageMeter() 
        # losses = AverageMeter()
        # acc = AverageMeter()

        # losses.update(loss.item(),input.shape[0])
        # acc.update(prec1.item(), input.shape[0])
        # batch_time.update(time.time() - end)

        # print(batch_time.avg,losses.avg,acc.avg)
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        比如某次acc=5/10=0.5，则val=0.5，n=10
        :param val: 值
        :param n: 个数
        :return: 
        """
        self.val = val 
        self.sum += val * n  
        self.count += n 
        self.avg = self.sum / self.count   
       
   
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    """计算top1或topk的acc"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        # pred保持概率最大的前五个值，target进行扩维，比如一个real_target=84的样本
        #          pred=[12,35,84,61,121]
        # expand_target=[84,84,84,84,84]
        # pred.eq()=>   [False,False,True,False,False]
        # 计算式top1时保证第一位True计算正确，top5时有一个True即算正确
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            # https://blog.csdn.net/xuan971130/article/details/109908149
            correct_k = correct[:k].contiguous().view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
   
 

