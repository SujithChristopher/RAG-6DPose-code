# Initially from https://github.com/usuyama/pytorch-unet (MIT License)
# Architecture slightly changed (removed some expensive high-res convolutions)
# and extended to allow multiple decoders
import torch
from torch import nn
import torchvision


def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )


class ResNetUNet(nn.Module):
    def __init__(self, n_class, feat_preultimate=64, n_decoders=1):
        super().__init__()

        #  shared encoder
        self.base_model = torchvision.models.resnet18(pretrained=True)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3])  # size=(N, 64, x.H/2, x.W/2)
        self.layer1 = nn.Sequential(*self.base_layers[3:5])  # size=(N, 64, x.H/4, x.W/4)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)

        #  n_decoders
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.decoders = [dict(
            layer0_1x1=convrelu(64, 64, 1, 0),
            layer1_1x1=convrelu(64, 64, 1, 0),
            layer2_1x1=convrelu(128, 128, 1, 0),
            layer3_1x1=convrelu(256, 256, 1, 0),
            layer4_1x1=convrelu(512, 512, 1, 0),
            conv_up3=convrelu(256 + 512, 512, 3, 1),
            conv_up2=convrelu(128 + 512, 256, 3, 1),
            conv_up1=convrelu(64 + 256, 256, 3, 1),
            conv_up0=convrelu(64 + 256, 128, 3, 1),
            conv_original_size=convrelu(128, feat_preultimate, 3, 1),
            conv_last=nn.Conv2d(feat_preultimate, n_class, 1),
        ) for _ in range(n_decoders)]

        # register decoder modules
        for i, decoder in enumerate(self.decoders):
            for key, val in decoder.items():
                setattr(self, f'decoder{i}_{key}', val)

    def forward(self, input, decoder_idx=None):
        if decoder_idx is None:
            assert len(self.decoders) == 1
            decoder_idx = [0]
        else:
            assert len(decoder_idx) == 1 or len(decoder_idx) == len(input)

        # encoder
        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        layers = [layer0, layer1, layer2, layer3, layer4]

        # decoders
        out = []
        for i, dec_idx in enumerate(decoder_idx):
            decoder = self.decoders[dec_idx]
            batch_slice = slice(None) if len(decoder_idx) == 1 else slice(i, i + 1)

            x = decoder['layer4_1x1'](layer4[batch_slice])
            x = self.upsample(x)
            for layer_idx in 3, 2, 1, 0:
                layer_slice = layers[layer_idx][batch_slice]
                layer_projection = decoder[f'layer{layer_idx}_1x1'](layer_slice)
                x = torch.cat([x, layer_projection], dim=1)
                x = decoder[f'conv_up{layer_idx}'](x)
                x = self.upsample(x)

            x = decoder['conv_original_size'](x)
            out.append(decoder['conv_last'](x))

        if len(decoder_idx) == 1:
            #  out: 1 x (B, C, H, W)
            return out[0]
        else:
            #  out: B x (1, C, H, W)
            return torch.stack(out)[:, 0]


import torch.nn as nn

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )   


class UNet(nn.Module):

    def __init__(self, input_channel):
        super().__init__()
                
        self.dconv_down1 = double_conv(input_channel, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)
        
        self.conv_last = nn.Conv2d(64, 3, 1)
        
        
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        x = self.dconv_down4(x)
        
        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)
        
        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        return out

class UNet_Encoder(nn.Module):

    def __init__(self, input_channel):
        super().__init__()
                
        self.dconv_down1 = double_conv(input_channel, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)        

        self.maxpool = nn.MaxPool2d(2)
        
        
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        x = self.dconv_down4(x)
        
        out = x
        
        return out


class UNet_Decoder(nn.Module):

    def __init__(self, input_channel):
        super().__init__()
                
        self.dconv_down1 = double_conv(input_channel, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv(input_channel, 256)
        self.dconv_up2 = double_conv(256, 128)
        self.dconv_up1 = double_conv(128, 64)
        
        self.conv_last = nn.Conv2d(64, 3, 1)
        
        
    def forward(self, x):
        x = self.upsample(x)        
        
        x = self.dconv_up3(x)
        x = self.upsample(x)        

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        return out


class UNet_Decoder_highdim(nn.Module):

    def __init__(self, input_channel):
        super().__init__()
                
        # self.dconv_down1 = double_conv(input_channel, 64)
        # self.dconv_down2 = double_conv(64, 128)
        # self.dconv_down3 = double_conv(128, 256)
        # self.dconv_down4 = double_conv(256, 512)        

        # self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up5 = double_conv(input_channel, 1024)
        self.dconv_up4 = double_conv(1024, 512)
        self.dconv_up3 = double_conv(512, 256)
        self.dconv_up2 = double_conv(256, 128)
        self.dconv_up1 = double_conv(128, 64)
        
        self.conv_last = nn.Conv2d(64, 3, 1)
        
        
    def forward(self, x):
        # x = self.upsample(x)  
        
        x = self.dconv_up5(x)
        # x = self.upsample(x)   
        
        x = self.dconv_up4(x)
        x = self.upsample(x)   
        
        x = self.dconv_up3(x)
        x = self.upsample(x)        

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        return out
    
class UNet_Decoder_highdim_matching(nn.Module):

    def __init__(self, input_channel, out_channel):
        super().__init__()
                
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up5 = double_conv(input_channel, 1024)
        self.dconv_up4 = double_conv(1024, 512)
        self.dconv_up3 = double_conv(512, 256)
        self.dconv_up2 = double_conv(256, 128)
        self.dconv_up1 = double_conv(128, 64)
        
        self.conv_last = nn.Conv2d(64, out_channel, 1)
        
        
    def forward(self, x):
        # x = self.upsample(x)  
        
        x = self.dconv_up5(x)
        # x = self.upsample(x)   
        
        x = self.dconv_up4(x)
        x = self.upsample(x)   
        
        x = self.dconv_up3(x)
        x = self.upsample(x)        

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        return out

class UNet_Decoder_highdim_matching_512(nn.Module):

    def __init__(self, input_channel, out_channel):
        super().__init__()
                
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up5 = double_conv(input_channel, 2048)
        self.dconv_up4 = double_conv(2048, 1024)
        self.dconv_up3 = double_conv(1024, 1024)
        self.dconv_up2 = double_conv(1024, 512)
        self.dconv_up1 = double_conv(512, 512)
        
        self.conv_last = nn.Conv2d(512, out_channel, 1)
        
        
    def forward(self, x):
        # x = self.upsample(x)  
        
        x = self.dconv_up5(x)
        # x = self.upsample(x)   
        
        x = self.dconv_up4(x)
        x = self.upsample(x)   
        
        x = self.dconv_up3(x)
        x = self.upsample(x)        

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        return out

class UNet_Decoder_highdim_matching_256(nn.Module):

    def __init__(self, input_channel, out_channel):
        super().__init__()
                
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up5 = double_conv(input_channel, 2048)
        self.dconv_up4 = double_conv(2048, 1024)
        self.dconv_up3 = double_conv(1024, 768)
        self.dconv_up2 = double_conv(768, 512)
        self.dconv_up1 = double_conv(512, 384)
        
        self.conv_last = nn.Conv2d(384, out_channel, 1)
        
        
    def forward(self, x):
        # x = self.upsample(x)  
        
        x = self.dconv_up5(x)
        # x = self.upsample(x)   
        
        x = self.dconv_up4(x)
        x = self.upsample(x)   
        
        x = self.dconv_up3(x)
        x = self.upsample(x)        

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        return out


class RealUNet_Decoder_highdim_matching_256(nn.Module):

    def __init__(self, input_channel, out_channel):
        super().__init__()
                
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up5 = double_conv(input_channel, 1536)
        self.dconv_up4 = double_conv(1536, 768)
        self.dconv_up3 = double_conv(768+256, 512)
        self.dconv_up2 = double_conv(512+64, 512)
        self.dconv_up1 = double_conv(512, 384)
        
        self.conv_last = nn.Conv2d(384, out_channel, 1)
        
        
    def forward(self, x, feat_list):
        layer0_112 = feat_list[0] # 112
        layer1_56 = feat_list[1] # 56
        # layer2_28 = feat_list[2] # 28
        
        x = self.dconv_up5(x) # 28-28
        
        x = self.dconv_up4(x) # 28-28
        x = self.upsample(x) # 28-56
        
        up3_x = torch.cat([x,layer1_56],dim=1) # 768 + 256
        x = self.dconv_up3(up3_x) # 56-56
        x = self.upsample(x)  #56-112      

        up2_x = torch.cat([x,layer0_112],dim=1) # 512 + 64
        x = self.dconv_up2(up2_x) # 112-112
        x = self.upsample(x)  # 112-224      
        
        x = self.dconv_up1(x) # 224-224
        
        out = self.conv_last(x)
        
        return out

class RealUNet_Decoder_highdim_matching_64(nn.Module):

    def __init__(self, input_channel, out_channel):
        super().__init__()
                
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up5 = double_conv(input_channel, 1024)
        self.dconv_up4 = double_conv(1024, 256)
        self.dconv_up3 = double_conv(256+256, 256)
        self.dconv_up2 = double_conv(256+64, 128)
        self.dconv_up1 = double_conv(128, 64)
        
        self.conv_last = nn.Conv2d(64, out_channel, 1)
        
        
    def forward(self, x, feat_list):
        layer0_112 = feat_list[0] # 112
        layer1_56 = feat_list[1] # 56
        # layer2_28 = feat_list[2] # 28
        
        x = self.dconv_up5(x) # 28-28
        
        x = self.dconv_up4(x) # 28-28
        x = self.upsample(x) # 28-56
        
        up3_x = torch.cat([x,layer1_56],dim=1) # 768 + 256
        x = self.dconv_up3(up3_x) # 56-56
        x = self.upsample(x)  #56-112      

        up2_x = torch.cat([x,layer0_112],dim=1) # 512 + 64
        x = self.dconv_up2(up2_x) # 112-112
        x = self.upsample(x)  # 112-224      
        
        x = self.dconv_up1(x) # 224-224
        
        out = self.conv_last(x)
        
        return out

class RealUNet_Decoder_highdim_matching_256_abl3(nn.Module):

    def __init__(self, input_channel, out_channel):
        super().__init__()
                
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up5 = double_conv(input_channel, 512)
        self.dconv_up4 = double_conv(512, 384)
        self.dconv_up3 = double_conv(384+256, 512)
        self.dconv_up2 = double_conv(384+64, 384)
        self.dconv_up1 = double_conv(384, 384)
        
        self.conv_last = nn.Conv2d(384, out_channel, 1)
        
        
    def forward(self, x, feat_list):
        layer0_112 = feat_list[0] # 112
        layer1_56 = feat_list[1] # 56
        # layer2_28 = feat_list[2] # 28
        
        x = self.dconv_up5(x) # 28-28
        
        x = self.dconv_up4(x) # 28-28
        x = self.upsample(x) # 28-56
        
        up3_x = torch.cat([x,layer1_56],dim=1) # 768 + 256
        x = self.dconv_up3(up3_x) # 56-56
        x = self.upsample(x)  #56-112      

        up2_x = torch.cat([x,layer0_112],dim=1) # 512 + 64
        x = self.dconv_up2(up2_x) # 112-112
        x = self.upsample(x)  # 112-224      
        
        x = self.dconv_up1(x) # 224-224
        
        out = self.conv_last(x)
        
        return out

class UNet_Decoder_highdim_matching_128(nn.Module):

    def __init__(self, input_channel, out_channel):
        super().__init__()
                
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up5 = double_conv(input_channel, 2048)
        self.dconv_up4 = double_conv(2048, 1024)
        self.dconv_up3 = double_conv(1024, 512)
        self.dconv_up2 = double_conv(512, 256)
        self.dconv_up1 = double_conv(256, 128)
        
        self.conv_last = nn.Conv2d(128, out_channel, 1)
        
        
    def forward(self, x):
        # x = self.upsample(x)  
        
        x = self.dconv_up5(x)
        # x = self.upsample(x)   
        
        x = self.dconv_up4(x)
        x = self.upsample(x)   
        
        x = self.dconv_up3(x)
        x = self.upsample(x)        

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        return out

class UNet_Decoder_highdim_mask(nn.Module):

    def __init__(self, input_channel):
        super().__init__()
                
        # self.dconv_down1 = double_conv(input_channel, 64)
        # self.dconv_down2 = double_conv(64, 128)
        # self.dconv_down3 = double_conv(128, 256)
        # self.dconv_down4 = double_conv(256, 512)        

        # self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up5 = double_conv(input_channel, 512)
        # self.dconv_up4 = double_conv(1024, 512)
        self.dconv_up3 = double_conv(512, 256)
        self.dconv_up2 = double_conv(256, 128)
        self.dconv_up1 = double_conv(128, 64)
        
        self.conv_last = nn.Conv2d(64, 1, 1)
        
        
    def forward(self, x):
        # x = self.upsample(x)  
        
        x = self.dconv_up5(x)
        x = self.upsample(x)   
        
        # x = self.dconv_up4(x)
        # x = self.upsample(x)   
        
        x = self.dconv_up3(x)
        x = self.upsample(x)        

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        return out


class Decoder_RegressPose(nn.Module):

    def __init__(self, input_channel):
        super().__init__()
        
        # self.dconv_rot_1 = double_conv(input_channel, input_channel)
        self.dconv_rot_2 = double_conv(input_channel, 1024)
        
        # self.dconv_trans_1 = double_conv(input_channel, input_channel)
        self.dconv_trans_2 = double_conv(input_channel, 1024)
        
        self.rot_head = nn.Linear(1024, 6)
        self.trans_head = nn.Linear(1024, 3)
        
        
    def forward(self, x, ):
        # x = self.upsample(x)  
        
        # x_rot = self.dconv_rot_1(x)
        x_rot = self.dconv_rot_2(x) # torch.Size([16, 1024, 28, 28])
        x_rot = x_rot.permute(0,2,3,1).reshape(x.size(0),-1, 1024)
        # print(x_rot.shape) # torch.Size([16, 784, 1024]) 
        rot_delta = self.rot_head(x_rot).mean(dim=1)
        # print(rot_delta.shape) # torch.Size([16, 6])
        
        # x_trans = self.dconv_trans_1(x)
        x_trans = self.dconv_trans_2(x)
        x_trans = x_trans.permute(0,2,3,1).view(x.size(0),-1, 1024)
        trans_delta = self.trans_head(x_trans).mean(dim=1).unsqueeze(-1)
        
        return rot_delta, trans_delta



class Decoder_LoRA_128(nn.Module):

    def __init__(self, input_channel, out_channel):
        super().__init__()
                
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up5 = double_conv(input_channel, 512)
        self.dconv_up4 = double_conv(512, 256)
        self.dconv_up3 = double_conv(256, 128)
        self.dconv_up2 = double_conv(128, 64)
        self.dconv_up1 = double_conv(64, 32)
        
        self.conv_last = nn.Conv2d(32, out_channel, 1)
        
        
    def forward(self, x):
        x = self.dconv_up5(x) # 28-28
        
        x = self.dconv_up4(x) # 28-28
        x = self.upsample(x) # 28-56
        
        x = self.dconv_up3(x) # 56-56
        x = self.upsample(x)  #56-112      

        x = self.dconv_up2(x) # 112-112
        x = self.upsample(x)  # 112-224      
        
        x = self.dconv_up1(x) # 224-224
        
        out = self.conv_last(x)
        
        return out

class Decoder_LoRA_mask(nn.Module):

    def __init__(self, input_channel):
        super().__init__()
                
        # self.dconv_down1 = double_conv(input_channel, 64)
        # self.dconv_down2 = double_conv(64, 128)
        # self.dconv_down3 = double_conv(128, 256)
        # self.dconv_down4 = double_conv(256, 512)        

        # self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up5 = double_conv(input_channel, 256)
        # self.dconv_up4 = double_conv(1024, 512)
        self.dconv_up3 = double_conv(256, 128)
        self.dconv_up2 = double_conv(128, 64)
        self.dconv_up1 = double_conv(64, 32)
        
        self.conv_last = nn.Conv2d(32, 1, 1)
        
        
    def forward(self, x):
        # x = self.upsample(x)  
        
        x = self.dconv_up5(x)
        x = self.upsample(x)   
        
        # x = self.dconv_up4(x)
        # x = self.upsample(x)   
        
        x = self.dconv_up3(x)
        x = self.upsample(x)        

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        return out

