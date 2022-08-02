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
# https://github.com/PaddleEdu/OCR-models-PaddlePaddle/blob/6f075dce5e53298e78c613a43ae4a7571a3b92c2/PSENet/models/backbone/resnet.py
def _initialize_weights(self):
    for m in self.sublayers():
        if isinstance(m, nn.Conv2D):
            """
            目标：Conv2D初始化，weight=normal(mean=0,std=math.sqrt(2. / n)),bias=0
            """
            n=m.weight.shape[0]*m.weight.shape[1]*m.weight.shape[2]
            v=np.random.normal(loc=0.,scale=np.sqrt(2./n),size=m.weight.shape).astype("float32")
            m.weight.set_value(v)
            if m.bias is not None:  # 如果有bias则全重设为0
                bias = paddle.zeros_like(m.bias)
                m.bias.set_value(bias)
        elif isinstance(m, nn.BatchNorm2D):
            """
            目标：BatchNorm2D初始化，weight=1,bias=0
            """
            # paddle所有原始BatchNorm2D，weight全=1，bias全=0
            # 故以下pp代码执行了但没有什么改变
            weight = paddle.ones_like(m.weight)
            m.weight.set_value(weight)

            bias = paddle.zeros_like(m.bias)
            m.bias.set_value(bias)
        elif isinstance(m, nn.Linear):
            """
            目标：Linear初始化，weight=normal(mean=0.0, std=0.01),bias=0
            """
            weight = paddle.normal(mean=0.0, std=0.01, shape=m.weight.shape)
            m.weight.set_value(weight)
            if m.bias is not None:  # 如果有bias则全重设为0
                bias = paddle.zeros_like(m.bias)
                m.bias.set_value(bias)
