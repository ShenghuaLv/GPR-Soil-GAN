import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler
from torchvision.models.resnet import resnet50
import torch.nn.functional as F
import math


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=False, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class GPR_PNet_Conv2(nn.Module):
    def __init__(self, in_size, out_size, bn_mom=0.1):
        super(GPR_PNet_Conv2, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(out_size, momentum=bn_mom),
            nn.ReLU(inplace=True),)
        self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(out_size, momentum=bn_mom),
            nn.ReLU(inplace=True),)

    def forward(self, inputs):
        outputs = self.conv2(self.conv1(inputs))
        return outputs


# down
class GPR_PNetDown(nn.Module):
    def __init__(self, in_size, out_size):
        super(GPR_PNetDown, self).__init__()

        self.down = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size=(3, 3), stride=(2, 2), padding=1),
                                  nn.BatchNorm2d(out_size),
                                  nn.ReLU(inplace=True))
        self.conv2 = GPR_PNet_Conv2(out_size, out_size)
    def forward(self, inputs):
        outputs1 = self.down(inputs)
        outputs = self.conv2(outputs1)
        return outputs

# up
class GPR_PNetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(GPR_PNetUp, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=3, stride=(2, 2), padding=1, output_padding=(1, 1))
        self.conv2 = GPR_PNet_Conv2(2 * out_size, out_size)

    def forward(self, inputs1, inputs2):
        outputs1 = self.up(inputs1)
        outputs2 = torch.cat([outputs1, inputs2], dim=1)
        outputs = self.conv2(outputs2)
        return outputs



class GPR_GAN_FC(nn.Module):
    def __init__(self, in_size, out_size):
        super(GPR_GAN_FC, self).__init__()
        self.fc = nn.Linear(in_size, out_size)

    def forward(self, inputs):
        outputs = self.fc(inputs)
        return outputs



class netB_Encoder(nn.Module):
    def __init__(self, input_channels):
        super(netB_Encoder, self).__init__()
        self.in_channels = input_channels
        # filters = [32, 64, 128, 128, 256]
        # filters = [32, 64, 128, 256, 512]
        # filters = [64, 128, 256, 512, 512]
        filters = [64, 128, 256, 512, 1024]
        filters_fc = [256, 128, 128, 256]
        self.Layer0 = GPR_PNet_Conv2(input_channels, filters[0])
        # 下采样层
        self.Layer1 = GPR_PNetDown(filters[0], filters[1])
        self.Layer2 = GPR_PNetDown(filters[1], filters[2])
        self.Layer3 = GPR_PNetDown(filters[2], filters[3])
        self.Layer4 = GPR_PNetDown(filters[3], filters[4])
        self.center1 = GPR_GAN_FC(filters_fc[0], filters_fc[1])
        self.center2 = GPR_GAN_FC(filters_fc[1], filters_fc[2])
        self.center3 = GPR_GAN_FC(filters_fc[2], filters_fc[3])


    def forward(self, x, is_drop_out, drop_out):
        # filters = [32, 64, 128, 128, 256]
        # filters = [32, 64, 128, 256, 512]
        filters = [64, 128, 256, 512, 1024]
        filters_fc = [256, 128, 128, 256]
        Layer0 = self.Layer0(x)
        Layer1 = self.Layer1(Layer0)
        Layer2 = self.Layer2(Layer1)
        Layer3 = self.Layer3(Layer2)
        Layer4 = self.Layer4(Layer3)
        # print(Layer4.shape)
        center0 = Layer4.view(-1, filters[4], filters_fc[0])
        center1 = self.center1(center0)
        center1 = F.dropout(center1, p=drop_out, training=is_drop_out, inplace=True)  # p张量被设置为0的概率,dropout
        center2 = self.center2(center1)
        center3 = self.center3(center2)
        center4 = center3.view(-1, filters[4], 16, 16)
        # print(center4.shape)

        return center4, Layer3, Layer2, Layer1, Layer0

class netH1_Decoder(nn.Module):
    def __init__(self, output_channels):
        super(netH1_Decoder, self).__init__()

        self.out_channels = output_channels
        # filters = [32, 64, 128, 128, 256]
        # filters = [32, 64, 128, 256, 512]
        filters = [64, 128, 256, 512, 1024]
        # 上采样层
        self.up1 = GPR_PNetUp(filters[4], filters[3])
        self.up2 = GPR_PNetUp(filters[3], filters[2])
        self.up3 = GPR_PNetUp(filters[2], filters[1])
        self.up4 = GPR_PNetUp(filters[1], filters[0])
        self.final1 = nn.Sequential(nn.Conv2d(filters[0], self.out_channels, 1),
                                    nn.ReLU(inplace=True))

    def forward(self, Layer4, Layer3, Layer2, Layer1, Layer0):
        up1 = self.up1(Layer4, Layer3)
        # print(up1.shape)
        up2 = self.up2(up1, Layer2)
        up3 = self.up3(up2, Layer1)
        up4 = self.up4(up3, Layer0)
        # up5 = self.up5(up4)
        final1 = self.final1(up4)
        final2 = F.interpolate(final1, scale_factor=1, mode='nearest')

        return final2


class netD_Discriminator(nn.Module):
    def __init__(self, input_channels):
        super(netD_Discriminator, self).__init__()

        self.ind_channels = input_channels

        self.main = nn.Sequential(
            # input is 512 x 16 x 16
            nn.Conv2d(self.ind_channels, self.ind_channels * 2, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(self.ind_channels * 2), nn.LeakyReLU(0.2, inplace=True),
            # state size. 1024 x 7 x 7
            nn.Conv2d(self.ind_channels * 2, self.ind_channels * 2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.ind_channels * 2), nn.LeakyReLU(0.2, inplace=True),
            # state size. 1 x 7 x 7
            nn.Conv2d(self.ind_channels * 2,   1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid())


    def forward(self, x):
        output = self.main(x)
        # print(output.shape)
        return output.view(-1, 1).squeeze(1)


# 对卷积层和BatchNorm层进行参数初始化
def weights_init_normal(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


data = torch.randn(8, 1, 256, 256)

# netB = netB_Encoder(1)
# center4, Layer3, Layer2, Layer1, Layer0 = netB(data, True, 0.3)
# print(center4.shape)
#
# netH = netH1_Decoder(1)
# outputs2 = netH(center4, Layer3, Layer2, Layer1, Layer0)
# print(outputs2.shape)

