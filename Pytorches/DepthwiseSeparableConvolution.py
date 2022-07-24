import torch
from torch import nn



class DSC(nn.Module):
    """
    深度可分离卷积：https://zhuanlan.zhihu.com/p/80041030 https://zhuanlan.zhihu.com/p/490685194
    先是depthwiseConv，本质上就是分组卷积，在深度可分离卷积中，分组卷积的组数=输入通道数=输出通道数，该部分通道数不变
    再是pointwisejConv，就是点卷积，该部分负责扩展通道数，所以其kernel_size=1，不用padding
    """
    def __init__(self, in_channel, out_channel, ksize=3,padding=1,bais=True):
        super(DSC, self).__init__()

        self.depthwiseConv = nn.Conv2d(in_channels=in_channel,
                                       out_channels=in_channel,
                                       groups=in_channel,
                                       kernel_size=ksize,
                                       padding=padding,
                                       bias=bais)
        self.pointwiseConv = nn.Conv2d(in_channels=in_channel,
                                       out_channels=out_channel,
                                       kernel_size=1,
                                       padding=0,
                                       bias=bais)

    def forward(self, x):
        out = self.depthwiseConv(x)
        out = self.pointwiseConv(out)
        return out

if __name__=="__main__":
    from torchsummary import summary
    dsc=DSC(in_channel=3,out_channel=8,ksize=3,padding=1,bais=False).cuda()
    summary(dsc,input_size=(3,48,48))


