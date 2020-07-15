from resnest.torch import resnest101
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
class fen(nn.Module):
  def __init__(self):
    super(fen,self).__init__()
    net=resnest101(pretrained=True)
    netlist=list(net.children())
    self.fe1=nn.Sequential(*netlist[0:4])#128
    self.fe2=nn.Sequential(*netlist[4])#256
    self.fe3=nn.Sequential(*netlist[5])#512
    self.fe4=nn.Sequential(*netlist[6])#1024
    self.fe5=nn.Sequential(*netlist[7])#2048
  def forward(self,x):
    fe1=self.fe1(x)
    fe2=self.fe2(fe1)
    fe3=self.fe3(fe2)
    fe4=self.fe4(fe3)
    fe5=self.fe5(fe4)
    return fe1,fe2,fe3,fe4,fe5
    
class seblock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(seblock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
        
class basicconv1(nn.Module):
  def __init__(self,in_c,out_c,convtype=nn.Conv2d):
    super(basicconv1,self).__init__()
    self.conv=convtype(in_c,out_c,3,1,1)
    self.bn=nn.BatchNorm2d(out_c)
    self.relu=nn.ReLU(inplace=True)
    self.se=seblock(out_c)
  def forward(self,x):
    xt=self.conv(x)
    return self.se(self.relu(self.bn(xt)))
    
class basicconv2(nn.Module):
  def __init__(self,in_c,out_c):
    super(basicconv2,self).__init__()
    self.conv=nn.Conv2d(in_c,out_c,4,2,1)
    self.bn=nn.BatchNorm2d(out_c)
    self.relu=nn.ReLU(inplace=True)
    self.se=seblock(out_c)
  def forward(self,x):
    xt=self.conv(x)
    return self.se(self.relu(self.bn(xt)))
    
class sodhead(nn.Module):
  def __init__(self):
    super(sodhead,self).__init__()
    #2-3,2-4,3-4

    #dilation 5,3,1
    #2+3
    self.convd1=basicconv2(256+512,512)
    self.convk1=basicconv1(512,256)
    self.convk2=basicconv1(256,128)
    self.convd2=nn.Sequential(
        nn.Conv2d(128,64,kernel_size=3,stride=1,dilation=5,padding=5),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True)
    )

    #2+4
    self.convd3=basicconv2(256+1024,512)
    self.convk3=basicconv1(512,256)
    self.convk4=basicconv1(256,128)
    self.convd4=nn.Sequential(
        nn.Conv2d(128,64,kernel_size=3,stride=1,dilation=3,padding=3),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True)
    )

    #3+4
    self.convd5=basicconv2(1024+512,512)
    self.convk5=basicconv1(512,256)
    self.convk6=basicconv1(256,128)
    self.convd6=nn.Sequential(
        nn.Conv2d(128,64,kernel_size=3,stride=1,dilation=1,padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True)
    )

    #sodhead
    self.sodf=nn.Conv2d(64*3,1,kernel_size=7,stride=1,padding=7)
    self.sodc=nn.Conv2d(64,1,kernel_size=5,stride=1,padding=5)
    self.sig=nn.Sigmoid()

  def forward(self,low,mid,high):
    #2,3,4

    lm=F.interpolate(mid,size=(low.shape[3],low.shape[2]))
    lm=torch.cat([low,lm],1)
    d1=self.convd1(lm)
    k1=self.convk1(d1)
    k2=self.convk2(k1)
    d2=self.convd2(k2)

    lowmap=F.interpolate(self.sodc(d2),size=(384,384))

    lh=F.interpolate(high,size=(low.shape[3],low.shape[2]))
    lh=torch.cat([low,lh],1)
    d3=self.convd3(lh)
    k3=self.convk3(d3)
    k4=self.convk4(k3)
    d4=self.convd4(k4)

    midmap=F.interpolate(self.sodc(d4),size=(384,384))

    mh=F.interpolate(high,size=(mid.shape[3],mid.shape[2]))
    mh=torch.cat([mid,mh],1)
    d5=self.convd5(mh)
    k5=self.convk5(d5)
    k6=self.convk6(k5)
    d6=self.convd6(k6)

    highmap=F.interpolate(self.sodc(d6),size=(384,384))
    d6=F.interpolate(d6,scale_factor=2)
    att=torch.cat([d2,d4,d6],1)
    att=self.sodf(att)
    #att=F.interpolate(att,size=(384,384))
    #BCEwithlogit
    return F.sigmoid(att),F.sigmoid(lowmap),F.sigmoid(midmap),F.sigmoid(highmap)
    
class rfrm(nn.Module):
  def __init__(self,ngf=64):
    super(rfrm,self).__init__()
    self.enc1=basicconv1(3+1,ngf)
    self.enc2=basicconv1(ngf,ngf*2)
    self.enc3=basicconv1(ngf*2,ngf*4)
    self.enc4=basicconv1(ngf*4,ngf*8)
    self.enc5=basicconv1(ngf*8,ngf*8)
    self.enc6=basicconv1(ngf*8,ngf*8)

    self.dec1=basicconv1(ngf*8,ngf*8,convtype=nn.ConvTranspose2d)
    self.dec2=basicconv1(ngf*16,ngf*8,convtype=nn.ConvTranspose2d)
    self.dec3=basicconv1(ngf*16,ngf*4,convtype=nn.ConvTranspose2d)
    self.dec4=basicconv1(ngf*8,ngf*2,convtype=nn.ConvTranspose2d)
    self.dec5=basicconv1(ngf*4,ngf,convtype=nn.ConvTranspose2d)
    #self.dec6=basicconv1(ngf,3,convtype=nn.ConvTranspose2d)
    self.dec6=nn.Sequential(
        nn.Conv2d(64,3,kernel_size=3,stride=1,padding=1)
    )

    self.tanh=nn.Tanh()
  def forward(self,img,att):
    att=F.interpolate(att,size=[384,384])
    aug=torch.cat([img,att],1)
    enc1=self.enc1(aug)
    enc1=F.interpolate(enc1,scale_factor=0.5)
    enc2=self.enc2(enc1)
    enc2=F.interpolate(enc2,scale_factor=0.5)
    enc3=self.enc3(enc2)
    enc4=self.enc4(enc3)
    enc5=self.enc5(enc4)
    enc6=self.enc6(enc5)

    dec1=self.dec1(enc6)

    sk2=F.interpolate(enc5,size=(dec1.shape[3],dec1.shape[2]))
    dec2=self.dec2(torch.cat([sk2,dec1],1))

    sk3=F.interpolate(enc4,size=(dec2.shape[3],dec2.shape[2]))
    dec3=self.dec3(torch.cat([sk3,dec2],1))

    sk4=F.interpolate(enc3,size=(dec3.shape[3],dec3.shape[2]))
    dec4=self.dec4(torch.cat([sk4,dec3],1))

    sk5=F.interpolate(enc2,size=(dec4.shape[3],dec4.shape[2]))
    dec5=self.dec5(torch.cat([sk5,dec4],1))

    dec6=self.dec6(dec5)

    dec6=F.interpolate(dec6,size=(384,384))

    return self.tanh(dec6)


class totalnet(nn.Module):
  def __init__(self):
    super(totalnet,self).__init__()
    self.fen=fen()
    for p in self.parameters(): 
      p.requires_grad = False
    self.sodhead=sodhead()
    self.rfrm=rfrm()
  def forward(self,x):
    fe1,fe2,fe3,fe4,fe5=self.fen(x)
    att,low,mid,high=self.sodhead(fe2,fe3,fe4)
    trans=self.rfrm(x,att)
    return trans,low,mid,high
    
class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=32, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0)]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)