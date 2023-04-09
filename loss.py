from re import L
import torch.nn as nn
from data.utils import *
#3 type of loss
'''
L(u,v)=||u-v||2/||u||2 + ||u-v||1/||u||1
u:input /\ unsampled ksapce data
v:output ksapce data with mask /\ 

Lac=lamda_img*L_img + lamda_grad*L_grad

'''
'''
input:Y(pred_1,pred_2,pred_u),X_p1,X_p2,X_u

'''
#input(B,C,H,W)
# fhsdghskgjh
def cal_loss(under_kspace,output_ksapce,mask_loss):
    u=output_ksapce*mask_loss #重建输出+mask
    v=under_kspace            #gt*loss_mask
    # epsilon=1e-2
    scaleL2 = torch.mean((torch.sum(torch.square(v)))).detach()
    scaleL1 = torch.mean((torch.sum(torch.abs(v)))).detach()
    # use output as fenmu
    lossl2=(((u-v)**2)/((u.detach())**2+scaleL2)).sum()/torch.sum(mask_loss[:,0,:,:]>0.001)
    lossl1=((((u-v).abs())/((u.detach()).abs()+scaleL1))).sum()/torch.sum(mask_loss[:,0,:,:]>0.001)
    L_tot = lossl1+lossl2
    
    return L_tot

#通过命令行参数 loss_eplison输入
def cal_loss_loupe(under_kspace,output_ksapce,mask_loss,epsilon):
    u=output_ksapce*mask_loss #重建输出+mask
    v=under_kspace            #gt*loss_mask
    # epsilon=1e-2
    # use output as fenmu
    lossl2=(((u-v)**2)/((u.detach())**2+epsilon)).sum()/torch.sum(mask_loss[:,0,:,:]>0.001)
    L_tot = lossl2
   
    return L_tot

def cal_loss_ssdu(under_kspace,output_ksapce,mask_loss):

    lossl1=nn.L1Loss()
    lossl2=nn.MSELoss()
    #k_space loss
    L_2=lossl2(under_kspace,output_ksapce*mask_loss)/lossl2(under_kspace,torch.zeros_like(under_kspace))
    L_1=lossl1(under_kspace,output_ksapce*mask_loss)/lossl1(under_kspace,torch.zeros_like(under_kspace))
    L_tot=L_1+L_2
    return L_tot


# former fenmu=gt
# def cal_loss(under_kspace,output_ksapce,mask_loss):
#     print('no clamp new loss')
#     u=output_ksapce*mask_loss #重建输出+mask
#     v=under_kspace            #gt*loss_mask
#     # epsilon=1e-8
#     scaleL2 = torch.mean((torch.sum(torch.square(v)))).detach()
#     scaleL1 = torch.mean((torch.sum(torch.abs(v)))).detach()
#     #做归一化
#     # u=torch.clamp(u,0,1)
#     # v=torch.clamp(v,0,1)
#     # u=u/(pseudo2real(u).max()+epsilon)
#     # v=v/(pseudo2real(v).max()+epsilon)
#     # print('mei gui yi hua loss')
#     lossl2=(((u-v)**2)/((v.detach())**2+scaleL2)).sum()/torch.sum(mask_loss[:,0,:,:]>0.001)
#     lossl1=((((u-v).abs())/((v.detach()).abs()+scaleL1))).sum()/torch.sum(mask_loss[:,0,:,:]>0.001)
#     L_tot = lossl1+lossl2
    
 
#     return L_tot

# loss NaN

# def cal_loss_que(under_kspace,output_ksapce,mask_loss):
#     print('new loss-clamp')
#     u=output_ksapce*mask_loss #重建输出+mask

#     v=under_kspace            #gt*loss_mask
#     epsilon=1e-8
#     #做归一化
#     # u=torch.clamp(u,0,1)
#     # v=torch.clamp(v,0,1)
#     # u=u/(pseudo2real(u).max()+epsilon)
#     # v=v/(pseudo2real(v).max()+epsilon)
#     # print(torch.max(pseudo2real(u)), torch.max(pseudo2real(v)))
#     fenzi=((u-v)**2)
#     # print('fenzi:',fenzi)
#     fenmu=((u.detach())**2+epsilon)
    
#     lossl2=(fenzi/fenmu).mean()
#     # L_tot=lossl2
#     lossl1=(((u-v).abs())/((u.detach()).abs()+epsilon)).mean()
#     L_tot = lossl1+lossl2

#     if torch.isnan(L_tot):
#         print('wrong')
#     # print('L_tot:',L_tot)
#     return L_tot




# def cal_loss_old(under_kspace,output_ksapce,mask_loss):
#     print('former loss')
#     lossl1=nn.L1Loss()
#     lossl2=nn.MSELoss()
#     #k_space loss
#     L_2=lossl2(under_kspace,output_ksapce*mask_loss)/lossl2(under_kspace,torch.zeros_like(under_kspace))
#     L_1=lossl1(under_kspace,output_ksapce*mask_loss)/lossl1(under_kspace,torch.zeros_like(under_kspace))
#     L_tot=L_1+L_2
#     return L_tot