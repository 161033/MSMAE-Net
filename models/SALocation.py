import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.seg_hrnet_config import get_cfg_defaults

def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
    return init_fun

class LastConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups, bias)
        self.mask_conv  = nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups, False)
        self.input_conv.apply(weights_init('kaiming'))
        torch.nn.init.constant_(self.mask_conv.weight, 1.0)

        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, input, mask):

        output = self.input_conv(input)
        if self.input_conv.bias is not None:
            output_bias = self.input_conv.bias.view(1, -1, 1, 1).expand_as(output)
        else:
            output_bias = torch.zeros_like(output)        

        with torch.no_grad():
            output_mask = self.mask_conv(mask)

        no_update_holes = output_mask == 0

        mask_sum = output_mask.masked_fill_(no_update_holes, 1.0)
        output_pre = (output - output_bias) / mask_sum + output_bias
        output = output_pre.masked_fill_(no_update_holes, 0.0)

        new_mask = torch.ones_like(output)
        new_mask = new_mask.masked_fill_(no_update_holes, 0.0)
        
        return output, new_mask

class Self_att(nn.Module):
    def __init__(self, in_channels, reduce_scale):
        super(Self_att, self).__init__()
        self.r = reduce_scale
        self.channel = in_channels * self.r * self.r

        self.mc = self.channel

        self.g = nn.Conv2d(in_channels=self.channel, out_channels=self.channel,
                           kernel_size=1, stride=1, padding=0)

        self.omiga = nn.Conv2d(in_channels=self.channel, out_channels=self.mc,
                               kernel_size=1, stride=1, padding=0)
        self.hi = nn.Conv2d(in_channels=self.channel, out_channels=self.mc,
                             kernel_size=1, stride=1, padding=0)
        self.W_s = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                             kernel_size=1, stride=1, padding=0)

        self.gamma_s = nn.Parameter(torch.ones(1))
        self.conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=18,
                                kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.conv_2 = nn.Conv2d(in_channels=18, out_channels=1, 
                                kernel_size=3, stride=1, padding=1)

        self.Last_conv1 = LastConv(3, 3, kernel_size=3, stride=2)
        self.Last_conv2 = LastConv(3, 1, kernel_size=3, stride=2)

    def forward(self, x, img):
        b, c, h, w = x.shape
        x1 = x.reshape(b, self.channel, h // self.r, w // self.r)
        # Gx
        gx = self.g(x1).view(b, self.channel, -1)
        gx = gx.permute(0, 2, 1)
        # omiga
        omiga_x = self.omiga(x1).view(b, self.mc, -1)
        omiga_xs = omiga_x.permute(0, 2, 1)
        hx = self.hi(x1).view(b, self.mc, -1)
        hxs = hx
        #As
        As_d = torch.matmul(omiga_xs, hxs)
        As = F.softmax(As_d, dim=-1)
        Hs = torch.matmul(As, gx)
        Hs = Hs.permute(0, 2, 1).contiguous()
        Hs = Hs.view(b, c, h, w)
        mask_feat = x + self.gamma_s * self.W_s(Hs)
        mask_feat = self.conv_1(mask_feat)
        mask_binary = mask_feat
        mask_binary = self.relu(mask_binary)
        mask_binary = self.conv_2(mask_binary)
        mask_binary = torch.sigmoid(mask_binary)
        mask_tmp = mask_binary.repeat(1, 3, 1, 1)
        mask_img = img * mask_tmp

        x, new_mask = self.Last_conv1(mask_img, mask_tmp)
        x, _ = self.Last_conv2(x, new_mask)
        mask_binary = mask_binary.squeeze(dim=1)
        return x, torch.sigmoid(mask_feat), mask_binary

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        
    def forward(self, x):       
        return x.view(x.size(0), -1)

class Classifer(nn.Module):
    def __init__(self, in_channels, output_channels):
        super(Classifer, self).__init__()
        self.pool = nn.Sequential(
                                  nn.AdaptiveAvgPool2d(1),
                                  Flatten()
                                )
        self.fc = nn.Linear(in_channels, output_channels, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.pool(x)
        feat = self.relu(feat)
        cls_res = self.fc(feat)
        return cls_res

class BranchCLS(nn.Module):
    def __init__(self, in_channels, output_channels):
        super(BranchCLS, self).__init__()
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),
                                  Flatten()
                                )
        self.fc = nn.Linear(18, output_channels, bias=True)
        self.bn = nn.BatchNorm1d(18, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.branch_cls = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=32, 
                                                  padding=1, kernel_size=3, stride=1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(in_channels=32, out_channels=18,
                                                  padding=1, kernel_size=3, stride=1),
                                        nn.ReLU(inplace=True), 
                                        )
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        feat = self.branch_cls(x)
        x = self.pool(feat)
        x = self.bn(x)
        cls_res = self.fc(x)
        cls_pro = self.leakyrelu(cls_res)
        zero_vec = -9e15*torch.ones_like(cls_pro)
        cls_pro = torch.where(cls_pro > 0, cls_pro, zero_vec)
        return cls_res, cls_pro, feat

class Smooth(nn.Module):
    def __init__(self, args, clip_dim=64, multi_feat=None):
        super(Smooth, self).__init__()
        ## obtain the dimensions. 
        feat1, feat2, feat3, feat4 = multi_feat

        self.smooth_s1 = nn.Sequential(
                                    nn.Conv2d(feat1, clip_dim, kernel_size=(1, 1), stride=(1, 1)),
                                    nn.Conv2d(clip_dim, clip_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))

        self.fpn1 = nn.Sequential(
            nn.Conv2d(clip_dim, clip_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(clip_dim),
            nn.ReLU())

        self.smooth_s2 = nn.Sequential(
                                    nn.Conv2d(feat2, clip_dim, kernel_size=(1, 1), stride=(1, 1)),
                                    nn.Conv2d(clip_dim, clip_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.fpn2 = nn.Sequential(
            nn.Conv2d(clip_dim, clip_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(clip_dim),
            nn.ReLU(),
            nn.Upsample(scale_factor=2))

        self.smooth_s3 = nn.Sequential(
                                    nn.Conv2d(feat3, clip_dim, kernel_size=(1, 1), stride=(1, 1)),
                                    nn.Conv2d(clip_dim, clip_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.fpn3 = nn.Sequential(
            nn.Conv2d(clip_dim, clip_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(clip_dim),
            nn.ReLU(),
            nn.Upsample(scale_factor=2))

        self.smooth_s4 = nn.Sequential(
                                    nn.Conv2d(feat4, clip_dim, kernel_size=(1, 1), stride=(1, 1)),
                                    nn.Conv2d(clip_dim, clip_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.fpn4 = nn.Sequential(
            nn.Conv2d(clip_dim, clip_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(clip_dim),
            nn.ReLU(),
            nn.Upsample(scale_factor=2))

class SALLoc(nn.Module):
    def __init__(self):
        super(SALLoc, self).__init__()
        self.crop_size = (256, 256)
        self.split_tensor_1 = torch.tensor([1, 3]).cuda()
        self.split_tensor_2 = torch.tensor([1, 2, 1, 3]).cuda()
        self.softmax_m = nn.Softmax(dim=1)
        FENet_cfg = get_cfg_defaults()
        dim = 64
        self.getmask = Self_att(dim, 4)
        self.Smooth = Smooth(dim, multi_feat=FENet_cfg['STAGE4']['NUM_CHANNELS'])

        self.branch_cls_level_1 = BranchCLS(317, 4)
        self.branch_cls_level_2 = BranchCLS(252, 3)
        self.branch_cls_level_3 = BranchCLS(216, 2)
        self.branch_cls_level_4 = BranchCLS(144, 2)

    def feature_resize(self, feat):
        s1, s2, s3, s4 = feat
        s1 = F.interpolate(s1, size=self.crop_size, mode='bilinear', align_corners=True)
        s2 = F.interpolate(s2, size=[i // 2 for i in self.crop_size], mode='bilinear', align_corners=True)
        s3 = F.interpolate(s3, size=[i // 4 for i in self.crop_size], mode='bilinear', align_corners=True)
        s4 = F.interpolate(s4, size=[i // 8 for i in self.crop_size], mode='bilinear', align_corners=True)
        return s1, s2, s3, s4

    def forward(self, feat, img):

        s1, s2, s3, s4 = self.feature_resize(feat)
        img = F.interpolate(img, size=self.crop_size, 
                            mode='bilinear', align_corners=True)

        feat_4 = self.Smooth.smooth_s4(s4)
        feat_4 = self.Smooth.fpn4(feat_4)
        feat_3 = self.Smooth.smooth_s3(s3)
        feat_3 = self.Smooth.fpn3(feat_3+feat_4)
        feat_2 = self.Smooth.smooth_s2(s2)
        feat_2 = self.Smooth.fpn2(feat_2+feat_3)
        feat_1 = self.Smooth.smooth_s1(s1)
        s1 = self.Smooth.fpn1(feat_1+feat_2)
        feat, mask, mask_binary = self.getmask(s1, img)
        feat = feat.clone().detach()
        conv1 = F.interpolate(feat, size=s1.size()[2:], mode='bilinear', align_corners=True)

        ## forth
        cls_4, pro_4, _ = self.branch_cls_level_4(s4)
        cls4      = self.softmax_m(pro_4)
        cls4_0 = torch.unsqueeze(cls4[:,0],1)
        cls4_1 = torch.unsqueeze(cls4[:,1],1)
        mask3 = torch.cat([cls4_0, cls4_1],axis=1)

        ## third
        s4F = F.interpolate(s4, size=s3.size()[2:], mode='bilinear', align_corners=True)
        s3_input = torch.cat([s4F, s3], axis=1)
        cls_3, pro_3, _ = self.branch_cls_level_3(s3_input)
        cls3      = self.softmax_m(pro_3)
        cls_3 = cls_3 + cls_3 * mask3
        cls3_0 = torch.unsqueeze(cls3[:,0],1)
        cls3_1 = torch.unsqueeze(cls3[:,1],1)

        mask2 = torch.cat([cls3_0, cls3_1, cls3_1],axis=1)

        ## second
        s3F = F.interpolate(s3_input, size=s2.size()[2:], mode='bilinear', align_corners=True)
        s2_input = torch.cat([s3F, s2], axis=1)
        cls_2, pro_2, _ = self.branch_cls_level_2(s2_input) 
        cls2      = self.softmax_m(pro_2)
        cls_2 = cls_2 + cls_2 * mask2
        cls2_0 = torch.unsqueeze(cls2[:,0],1)
        cls2_1 = torch.unsqueeze(cls2[:,1],1)
        cls2_2 = torch.unsqueeze(cls2[:,2],1)

        mask1 = torch.cat([cls2_0, cls2_1, cls2_2,cls2_2], axis=1)        # 3 editing

        s2F = F.interpolate(s2_input, size=s1.size()[2:], mode='bilinear', align_corners=True)
        s1_input = torch.cat([s2F, s1, conv1], axis=1)
        cls_1, pro_1, _ = self.branch_cls_level_1(s1_input) 
        cls_1 = cls_1 + cls_1 * mask1

        mask = mask.squeeze(dim=1)
        return mask, mask_binary, cls_4, cls_3, cls_2, cls_1
