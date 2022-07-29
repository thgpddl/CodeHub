# 

# pytorch版本
def _initialize_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):        # weight=N(0,)，bias=0
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            # .data没有requires_grad=True
            m.weight.data.normal_(0, math.sqrt(2. / n))
            # 经证明，normal_结果和weight本身没有关系，只是生成与weight.shape一样的正态分布随机数
            # a=torch.ones_like(m.weight.data)
            # nn.init.normal_(m.weight.data,mean=0,std=math.sqrt(2. / n))   # 同上
            if m.bias is not None: 
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d): # weight=1，bias=0
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):      # weight=N(0,0.01)，bias=0
            n = m.weight.size(1)
            m.weight.data.normal_(0, 0.01)
            if m.bias is not None:
                m.bias.data.zero_()
 
# pp版本
def _initialize_weights(self):
    for m in self.sublayers():
        if isinstance(m, nn.Conv2D):
            """
            目标：Conv2D初始化，weight=normal(mean=0,std=math.sqrt(2. / n)),bias=0
            """
            n = m._kernel_size[0] * m._kernel_size[1] * m._out_channels
            weight=paddle.normal(mean=0.0, std=math.sqrt(2. / n), shape=m.weight.shape)
            d = collections.OrderedDict()
            d['weight']=weight
            m.set_dict(d)
            if m.bias is not None:  # 如果有bias则全重设为0
                bias = paddle.zeros_like(m.bias)
                d = collections.OrderedDict()
                d['bias'] = bias
                m.set_dict(d)
        elif isinstance(m, nn.BatchNorm2D):
            """
            目标：BatchNorm2D初始化，weight=1,bias=0
            """
            # paddle所有原始BatchNorm2D，weight全=1，bias全=0
            # 故以下pp代码执行了但没有什么改变
            weight = paddle.ones_like(m.weight)
            d = collections.OrderedDict()
            d['weight'] = weight
            m.set_dict(d)
            bias = paddle.zeros_like(m.bias)
            d = collections.OrderedDict()
            d['bias'] = bias
            m.set_dict(d)
        elif isinstance(m, nn.Linear):
            """
            目标：Linear初始化，weight=normal(mean=0.0, std=0.01),bias=0
            """
            weight = paddle.normal(mean=0.0, std=0.01, shape=m.weight.shape)
            d = collections.OrderedDict()
            d['weight'] = weight
            m.set_dict(d)
            if m.bias is not None:  # 如果有bias则全重设为0
                bias = paddle.zeros_like(m.bias)
                d = collections.OrderedDict()
                d['bias'] = bias
                m.set_dict(d)
