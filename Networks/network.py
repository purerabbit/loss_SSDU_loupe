
from .unet.unet_model import UNet#用此方式可以实现包的导入

#从不同文件夹下导入包
import torch
import torch.nn as nn
import torch.nn.functional as F
from mri_tools import  ifft2,fft2
from .cascade import CascadeMRIReconstructionFramework
from .memc_loupe import Memc_LOUPE
from .memc_loupe2 import Memc_LOUPE2
from .umemc_loupe import UMemc_LOUPE
# from .total_mask_loupe import TOTAL_LOUPE
from data.utils import *
import scipy.io as sio
#从不同文件夹下导入包

class ParallelNetwork(nn.Module):
   
    def __init__(self, num_layers, rank,slope,sample_slope,sparsity,method):
        super(ParallelNetwork, self).__init__()
        self.num_layers = num_layers
        self.rank = rank
        self.net = CascadeMRIReconstructionFramework(
            n_cascade=5  #the formor is 5
        )
        if method=="undermask":
            input_shape_one=[1,1,1,232]  #只对高频采样
            self.Under_LOUPE_Model = UMemc_LOUPE(input_shape_one, slope=slope, sample_slope=sample_slope, device=self.rank, sparsity=0.17) #剩余部分学习比例(256*0.25-24)/(256-232)
        elif method=="selectmask1":
            input_shape_select1=[1,1,256,64] #对采样到的24条线进行采样
            self.Select_LOUPE_Model = Memc_LOUPE(input_shape_select1, slope=slope, sample_slope=sample_slope, device=self.rank, sparsity=0.6)
        # elif method=="selectmask2":
        #     input_shape_select2=[1,2,256,64] #实部虚部不同时针对全局进行分析
        #     self.Select_LOUPE_Model2 = Memc_LOUPE2(input_shape_select2, slope=slope, sample_slope=sample_slope, device=self.rank, sparsity=0.6)
        else:
            print('now method is baseline')
    def get_submask(self,getmask):
        '''
        功能：得到采样到部分矩阵 以及对应坐标
        输入:模拟欠采的mask
        输出:onemask:对应到采样到部分矩阵大小的矩阵   b:采样到部分数据的坐标(tuple b[0]对应横坐标    b[1]对应纵坐标)
        '''
        a=getmask[:,0,:,:]  # B H W
        b=((a == 1).nonzero(as_tuple=True))  #b[0]对应的非零值行坐标  b[1]对应的非零值列坐标
        onemask=torch.ones(1,a.shape[0],a.shape[1],len(torch.unique(b[2]))).to(self.rank)
        return onemask,b  # onemask.shape->B H select_W     b.len->3

    def recovery_mask(self,mask,sub_mask,b):
        recomask=mask.clone()
        sub_real=sub_mask[:,0,:,:]
        sub_img=sub_mask[:,1,:,:]
        list_mask_real=sub_real.reshape(-1)
        list_mask_img=sub_img.reshape(-1)
        # n=0
        recomask[b[0],0,b[1],b[2]]=list_mask_real
        recomask[b[0],1,b[1],b[2]]=list_mask_img

        return recomask

    def forward(self,mask,gt,method,mode,gdc_mask,gloss_mask):
        # dc_mask=None
        
        if mode=='train':
            if method=="undermask":
                kspace_shape = torch.ones((1,1,256,232)).to(self.rank) #去除中心部分
                maskloupe = self.Under_LOUPE_Model(kspace_shape,method=method)
                maskloupe=maskloupe.repeat(1,2,1,1) #将1 channel 变成2 channel
                maskl1=maskloupe[:,:,:,:116]
                maskl2=maskloupe[:,:,:,116:]
                mask_center = torch.ones((1,2,256,24)).to(self.rank)
                mask_total=torch.cat((maskl1,mask_center,maskl2),-1)
                mask=mask_total
                loss_mask=mask*gloss_mask
                dc_mask=mask*gdc_mask
            
            elif method=='selectmask1':
                onemask,b = self.get_submask(mask)
                sub_dc_mask = self.Select_LOUPE_Model(onemask,method=method)  #B H select_W
                sub_dc_mask=sub_dc_mask.repeat(1,2,1,1)
                mask_dc = self.recovery_mask(mask,sub_dc_mask,b)
                dc_mask = mask_dc
                loss_mask = mask - mask_dc

            # elif method=='selectmask2': #划分mask部分两个通道分开学 效果应该会好
            #     # print('selectmask')
            #     onemask,b = self.get_submask(mask)
            #     sub_dc_mask = self.Select_LOUPE_Model2(onemask,method=method)  #B H select_W
            #     mask_dc = self.recovery_mask(mask,sub_dc_mask,b)
            #     dc_mask = mask_dc
            #     loss_mask = mask - mask_dc
            elif method=="baseline":
                # baseline使用指定的mask进行实现--读取进来的只是用来select的
                dc_mask=gdc_mask*mask
                loss_mask=gloss_mask*mask
                mask=mask

        else:
            if method=='undermask': #如果是学原始欠采mask的话，实际测试的时候也用学到的mask
                kspace_shape = torch.ones((1,1,256,232)).to(self.rank) #去除中心部分
                maskloupe = self.Under_LOUPE_Model(kspace_shape,method,option=False)
                maskloupe=maskloupe.repeat(1,2,1,1) #将1 channel 变成2 channel
                maskl1=maskloupe[:,:,:,:116]
                maskl2=maskloupe[:,:,:,116:]
                mask_center = torch.ones((1,2,256,24)).to(self.rank)
                mask_total=torch.cat((maskl1,mask_center,maskl2),-1)
                mask=mask_total

            loss_mask=dc_mask=mask
        
        k0_recon=fft2(gt)*dc_mask #原始mask 用于计算dc
        im_recon=ifft2(k0_recon)  #输入网络的初始图像
        output_img=self.net(im_recon ,dc_mask,k0_recon)
        return  output_img,loss_mask,dc_mask,mask



