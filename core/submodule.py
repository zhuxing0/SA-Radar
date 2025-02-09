import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class BasicConv(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, bn=True, relu=True, **kwargs):
        super(BasicConv, self).__init__()

        self.relu = relu
        self.use_bn = bn
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.relu:
            x = nn.LeakyReLU()(x)#, inplace=True)
        return x

class hourglass_v2(nn.Module):
    def __init__(self, in_channels):
        super(hourglass_v2, self).__init__()

        self.conv1 = nn.Sequential(BasicConv(in_channels, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))
                                    
        self.conv2 = nn.Sequential(BasicConv(in_channels*2, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))                             

        self.conv3 = nn.Sequential(BasicConv(in_channels*4, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=(3,3,3),
                                             padding=(1,1,1), stride=(2,2,1), dilation=1),
                                   BasicConv(in_channels*6, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=(3,3,3),
                                             padding=(1,1,1), stride=(1,1,1), dilation=1)) 

        self.conv4 = nn.Sequential(BasicConv(in_channels*6, in_channels*8, is_3d=True, bn=True, relu=True, kernel_size=(3,3,3),
                                             padding=(1,1,1), stride=(2,2,1), dilation=1),
                                   BasicConv(in_channels*8, in_channels*8, is_3d=True, bn=True, relu=True, kernel_size=(3,3,3),
                                             padding=(1,1,1), stride=(1,1,1), dilation=1)) 

        self.conv4_up = BasicConv(in_channels*8, in_channels*6, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 3), padding=(1, 1, 1), stride=(2, 2, 1))
        
        self.conv3_up = BasicConv(in_channels*6, in_channels*4, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 3), padding=(1, 1, 1), stride=(2, 2, 1))

        self.conv2_up = BasicConv(in_channels*4, in_channels*2, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv1_up = BasicConv(in_channels*2, 8, deconv=True, is_3d=True, bn=False,
                                  relu=False, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.agg_3 = nn.Sequential(BasicConv(in_channels*12, in_channels*6, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*6, in_channels*6, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*6, in_channels*6, is_3d=True, kernel_size=3, padding=1, stride=1),)
        
        self.agg_2 = nn.Sequential(BasicConv(in_channels*8, in_channels*4, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),)

        self.agg_1 = nn.Sequential(BasicConv(in_channels*4, in_channels*2, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1))

    def forward(self, x):
        '''
            input: x, torch.Size([4, 32, 256, 512, 12]
            output: torch.Size([4, 8, 256, 512, 12])
        '''
        conv1 = self.conv1(x) # in_channels*2 (128, 256, 6)

        conv2 = self.conv2(conv1) # in_channels*4 (64, 128, 3)
        
        conv3 = self.conv3(conv2) # in_channels*6 (32, 64, 3)

        conv4 = self.conv4(conv3) # in_channels*8 (16, 32, 3)

        conv4_up = self.conv4_up(conv4) # in_channels*6 (32, 64, 3)
        conv3 = torch.cat((conv4_up, conv3), dim=1) # in_channels*12
        conv3 = self.agg_3(conv3) # in_channels*6

        conv3_up = self.conv3_up(conv3) # in_channels*4 (64, 128, 3)
        conv2 = torch.cat((conv3_up, conv2), dim=1) # in_channels*8
        conv2 = self.agg_2(conv2) # in_channels*4

        conv2_up = self.conv2_up(conv2) # in_channels*2
        conv1 = torch.cat((conv2_up, conv1), dim=1) # in_channels*4
        conv1 = self.agg_1(conv1) # in_channels*2

        conv = self.conv1_up(conv1) # 8

        return conv