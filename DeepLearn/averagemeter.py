class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
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
    
   
 
# batch_time=AverageMeter() 
# losses = AverageMeter()
# acc = AverageMeter()

# losses.update(loss.item(),input.shape[0])
# acc.update(prec1.item(), input.shape[0])
# batch_time.update(time.time() - end)

# print(batch_time.avg,losses.avg,acc.avg)
