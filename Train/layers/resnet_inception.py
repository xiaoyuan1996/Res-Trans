import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import math,sys
sys.path.append("..")
import config
from layers.net_utils import FC, MLP, LayerNorm

# 输入图片大小：W x W
# Filter大小：F x F
# Stride：S
# Padding ：P
# 输出图片：N x N
# N = ( W - F + 2*P )/S + 1

# 用于ResNet18和34的残差块，用的是2个3*3的卷积
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride!=1 or in_channels!=self.expansion*out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*self.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(self.expansion*out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out+self.shortcut(x))
        return out

# 用于ResNet50,101和152的残差块，用的是1*1+3*3+1*1的卷积
class BottleNeck(nn.Module):
    # 前边1*1和3*3卷积的filter个数相等，最后1*1卷积是其expansion倍
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, self.expansion*out_channels, kernel_size=1, padding=0, stride=1)
        self.bn3 = nn.BatchNorm2d(self.expansion*out_channels)

        self.shortcut = nn.Sequential()
        if stride!=1 or in_channels != self.expansion*out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion*out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(self.expansion*out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=4):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, padding=3, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        # self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # 3层10752 4层24576
        self.linear = nn.Linear(256*4*4, num_classes)
        self._initialize_weights()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, planes, stride))
            # print(self.in_channels, planes,)
            self.in_channels = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        # # out = self.layer3(out)
        # # out = self.layer4(out)
        # print(out.shape)
        #
        out = F.avg_pool2d(out, 4)
        # print(out.shape)
        #
        # out = out.view(out.size(0), -1)
        # out = F.dropout(out,0.4)
        # # print(out.shape)
        # # print(out.shape)
        #
        # out = self.linear(out)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet3():
    return ResNet(BasicBlock, [1,1,1,1])

def ResNet4():
    return ResNet(BottleNeck, [1,1,1,1])

def ResNet101():
    return ResNet(BottleNeck, [3,4,23,3])

def ResNet152():
    return ResNet(BottleNeck, [3,8,36,3])


def load_pre_model(net):
    resnet18 = torch.load("resnet18.pth")

    # 建立映射字典
    for k,v in net.named_parameters():
        if k in resnet18.keys():
            v.data = resnet18[k].data


# res = ResNet50()
# # print(res)
# # in_demo = Variable(torch.zeros(1,3,224,224))
# # out_demo = res(in_demo)
# # print(out_demo.shape)
# load_pre_model(res)
class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

from torchvision.models import inception_v3
class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features, conv_block=None):
        super(InceptionA, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = conv_block(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = conv_block(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = conv_block(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(96, 96, kernel_size=3, padding=1)

        self.branch_pool = conv_block(in_channels, pool_features, kernel_size=1)

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.m = ResNet3()
        self.incep = InceptionA(128,16)

        self.linear = nn.Linear(240*8*8,4)


    def forward(self, x):
        x = self.m(x)
        # print(x.shape)
        x = self.incep(x)
        # print(x.shape)
        # x = F.max_pool2d(x, kernel_size=3, stride=2)
        # print(x.shape)
        x = F.dropout(x,p=0.4)
        x = F.softmax(self.linear(x.view(x.shape[0],-1)),dim=-1)
        return x

if __name__=="__main__":
    # m = InceptionA(128,32)
    m = model()
    in_demo = Variable(torch.zeros(10,3,256,256))
    out_demo = m(in_demo)
    print(out_demo.shape)

