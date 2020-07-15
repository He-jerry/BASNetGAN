#! /usr/bin/env python
import argparse
import os
import numpy as np
import math
import itertools
import sys
import torch
import torch.nn as nn
import torchvision
from loss import VGGLoss
torch.cuda.empty_cache()
from dataset import ImageDataset
from network import totalnet,PixelDiscriminator
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn.functional as F
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
# Loss functions
#new code
criterion_GAN= torch.nn.MSELoss()
criterion_pixelwise = torch.nn.L1Loss()#L2 loss
criterion_bce=torch.nn.BCEWithLogitsLoss()
criterion_vgg=VGGLoss()

gen1 =totalnet()
discriminator1 = PixelDiscriminator()


gen1 = gen1.cuda()
discriminator1 = discriminator1.cuda()
criterion_GAN.cuda()
criterion_pixelwise.cuda()
criterion_bce.cuda()
criterion_vgg.cuda()
#7.7

#gen1.apply(weights_init_normal)
#gen2.apply(weights_init_normal)
#discriminator1.apply(weights_init_normal)
#discriminator2.apply(weights_init_normal)
# Optimizers
optimizer_G1 = torch.optim.SGD(filter(lambda p: p.requires_grad, gen1.parameters()), lr=0.002, momentum=0.9)
optimizer_D1 = torch.optim.SGD(discriminator1.parameters(), lr=0.0005, momentum=0.9)
#optimizer_T = torch.optim.Adam(netparam, lr=0.00005, betas=(0.5, 0.999))
# Configure dataloaders


trainloader = DataLoader(
    ImageDataset(transforms_=None),
    batch_size=2,
    shuffle=False,drop_last=True
)
print("data length:",len(trainloader))
Tensor = torch.cuda.FloatTensor
eopchnum=70#10+40+10+40+60
print("start training")
for epoch in range(0, eopchnum):
  print("epoch:",epoch)
  iteration=0
  for i, total in enumerate(trainloader):
    
    iteration=iteration+1
    # Model inputs
    real_img = total["mix"]
    real_trans=total["trans"]
    real_mask = total["mask"]
    real_img=real_img.cuda()
    real_trans=real_trans.cuda()
    real_mask=real_mask.cuda()

    # Adversarial ground truths
    valid = Variable(Tensor(np.ones((2, 1,384,384))), requires_grad=False)
    fake = Variable(Tensor(np.zeros((2, 1,384,384))), requires_grad=False)
    print("batch:%3d,iteration:%3d"%(epoch+1,iteration))
    # ------------------
    #  Train Generators1,2
    # ------------------

    optimizer_G1.zero_grad()
    trans,low,mid,high=gen1(real_img)
    loss_map=criterion_bce(low,real_mask)+criterion_bce(mid,real_mask)+criterion_bce(high,real_mask)
    loss_rfrm=criterion_vgg(trans,real_trans)/7
    loss_G1=loss_map+loss_rfrm
    print("loss_map:%3f  loss_rfrm:%3f"%(loss_map.item(),loss_rfrm.item()))
    
    loss_G1.backward()

    optimizer_G1.step()


    
    
    # ---------------------
    #  Train Discriminator1,2
    # --------------------- 

    optimizer_D1.zero_grad()
    pred_real=discriminator1(real_trans)
    loss_real=criterion_GAN(pred_real,valid)
    
    pred_fake=discriminator1(trans.detach())
    loss_fake=criterion_GAN(pred_fake,fake)
    
    loss_D1=loss_real+loss_fake
    
    loss_D1.backward()
    optimizer_D1.step()
    print("dis1:%3f"%(loss_D1.item()))

    
    
    
    
  if(epoch%10==0):
    torch.save(gen1,"generator1_refine_%3d.pth"%epoch)
    torch.save(discriminator1,'discriminator1_refine_%3d.pth'%epoch)



    