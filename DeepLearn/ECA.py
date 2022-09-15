# paddle实现
class eca_layer(nn.Layer):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
    """

    def __init__(self, channel, gamma=2, b=1):
        super(eca_layer, self).__init__()

        t = int(abs((np.log2(channel) + b) / gamma))
        k_size = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.conv = nn.Conv1D(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias_attr=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # shape=1,16,40,40
        y = self.avg_pool(x)  # shape=1,16,1,1

        y = y.squeeze(-1)  # shape=1,16,1
        y = paddle.transpose(y, [0, 2, 1])  # shape=1,1,16
        y = self.conv(y)  # shape=1,1,16
        y = paddle.transpose(y, [0, 2, 1])  # shape=1,16,1
        y = y.unsqueeze(-1)  # shape=1,16,1,1

        y = self.sigmoid(y)

        return x * y.expand_as(x)
        
        
# torch实现
class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)
