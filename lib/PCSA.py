import torch
import torch.nn as nn

class PCSA(nn.Module):
    def __init__(self, channels, c2=None, factor=8):
        super(PCSA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3_1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

        self.conv3x3_2 = nn.Conv2d(channels, channels , kernel_size=3, stride=1, padding=1)
        self.bn=nn.BatchNorm2d(channels,affine=False)
        self.cgp = nn.AdaptiveAvgPool2d((None,1))

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x0 = group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid()
        x1 = self.gn(x0)
        x2 = self.conv3x3_1(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x1.reshape(b * self.groups, c // self.groups, -1)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x2.reshape(b * self.groups, c // self.groups, -1)
        weights1 = (torch.matmul(x11, x22) + torch.matmul(x21, x12)).reshape(b * self.groups, 1, h, w)

        x3 = self.bn(x0.reshape(b, c, h, w))
        x4 = self.conv3x3_2(x)
        x31 = self.softmax(self.cgp(x3.permute(0, 2, 3,1)).permute(0, 3, 1,2).reshape(b , -1, h*w))
        x32 = x3.reshape(b,c,h*w).permute(0, 2,1)
        x41 = self.softmax(self.cgp(x4.permute(0, 2, 3,1)).permute(0, 3, 1,2).reshape(b , -1, h*w))
        x42 = x4.reshape(b,c,h*w).permute(0, 2,1)
        weights2 = (torch.matmul(x31, x42) + torch.matmul(x41, x32)).reshape(b , c, -1, 1)
        return (group_x * weights1.sigmoid()).reshape(b, c, h, w)+(x * weights2.sigmoid())