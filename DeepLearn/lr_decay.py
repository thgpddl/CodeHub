
# Pytorch版本
from math import cos, pi
def adjust_learning_rate(optimizer, epoch, iteration, num_iter):
    """
    学习率衰减策略
    Args:
        optimizer: 优化器，将读取它的lr，然后改变其lr
        epoch: 当前是第epoch次
        iteration: 当前是epoch中第iter次迭代（第iter次从loader读取数据）
        num_iter: 每个epoch最大iter次数（即每个epoch中含有num_iter个迭代次数）

    Returns: 无返回，是直接改变optimizer中的lr

    """

    warmup_epoch = 5 if args.warmup else 0  # 是否热启动，是则前5epoch执行lr热启动
    warmup_iter = warmup_epoch * num_iter   # warmup_iter是5次epoch总迭代书
    current_iter = iteration + epoch * num_iter # 从epoch=0以来，累计经过了current_iter次迭代，即当前迭代次数
    max_iter = args.epochs * num_iter   # args.epochs是总epochs数，即max_iter本次训练总共的迭代数

    if args.lr_decay == 'step':
        lr = args.lr * (args.gamma ** ((current_iter - warmup_iter) // (max_iter - warmup_iter)))
    elif args.lr_decay == 'cos':
        lr = args.lr * (1 + cos(pi * (current_iter - warmup_iter) / (max_iter - warmup_iter))) / 2
    elif args.lr_decay == 'linear':
        lr = args.lr * (1 - (current_iter - warmup_iter) / (max_iter - warmup_iter))
    elif args.lr_decay == 'schedule':
        count = sum([1 for s in args.schedule if s <= epoch])
        lr = args.lr * pow(args.gamma, count)
    else:
        raise ValueError('Unknown lr mode {}'.format(args.lr_decay))

    if epoch < warmup_epoch:    # 热启动阶段，在前5epoch阶段，lr会线性从0增长至预先设定的lr
        lr = args.lr * current_iter / warmup_iter

    for param_group in optimizer.param_groups:  # 将改变后的lr写入优化器，在训练是则会使用新的lr
        param_group['lr'] = lr
  
 
# Paddle版本
from math import cos, pi
def adjust_learning_rate(optimizer, epoch, iteration, num_iter):
    """
    学习率衰减策略
    Args:
        optimizer: 优化器，将读取它的lr，然后改变其lr
        epoch: 当前是第epoch次
        iteration: 当前是epoch中第iter次迭代（第iter次从loader读取数据）
        num_iter: 每个epoch最大iter次数（即每个epoch中含有num_iter个迭代次数）

    Returns: 无返回，是直接改变optimizer中的lr

    """
    
    warmup_epoch = 5 if args.warmup else 0  # 是否热启动，是则前5epoch执行lr热启动
    warmup_iter = warmup_epoch * num_iter   # warmup_iter是5次epoch总迭代书
    current_iter = iteration + epoch * num_iter # 从epoch=0以来，累计经过了current_iter次迭代，即当前迭代次数
    max_iter = args.epochs * num_iter   # args.epochs是总epochs数，即max_iter本次训练总共的迭代数
    # 以上参数因为很多下降都是从预定lr变化到0，也即最开始（warmup之后）lr就是预设值，然后每次迭代lr改变一次，
    # 到最后依次迭代lr几乎下降到0，所以我们需要知道总的迭代数和当前迭代数，方便计算变化

    if args.lr_decay == 'step':
        lr = args.lr * (args.gamma ** ((current_iter - warmup_iter) // (max_iter - warmup_iter)))
    elif args.lr_decay == 'cos':
        lr = args.lr * (1 + cos(pi * (current_iter - warmup_iter) / (max_iter - warmup_iter))) / 2
    elif args.lr_decay == 'linear':
        lr = args.lr * (1 - (current_iter - warmup_iter) / (max_iter - warmup_iter))
    elif args.lr_decay == 'schedule':
        count = sum([1 for s in args.schedule if s <= epoch])
        lr = args.lr * pow(args.gamma, count)
    else:
        raise ValueError('Unknown lr mode {}'.format(args.lr_decay))

    if epoch < warmup_epoch:    # 热启动阶段，在前5epoch阶段，lr会线性从0增长至预先设定的lr
        lr = args.lr * current_iter / warmup_iter

    for param_group in optimizer.param_groups:  # 将改变后的lr写入优化器，在训练是则会使用新的lr
        param_group['lr'] = lr
