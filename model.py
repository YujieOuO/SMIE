from config import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat,rearrange
import random

def init_weights(m):
    class_name=m.__class__.__name__

    if "Linear" in class_name:
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0.0)
    
class GlobalDiscriminator(nn.Module):
    
    @ex.capture
    def __init__(self, in_feature):
        super().__init__()
        self.l0 = nn.Linear(in_feature, 1024)
        self.l1 = nn.Linear(1024, 512)
        self.l2 = nn.Linear(512, 1)
        # self.apply(init_weights)

    def forward(self, visual, language):

        x = torch.cat((visual, language), dim=-1)
        out = F.relu(self.l0(x))
        out = F.relu(self.l1(out))
        out = self.l2(out)
        return out

class MI(nn.Module):

    @ex.capture
    def __init__(self, visual_size, language_size):
        super(MI,self).__init__()

        self.global_D = GlobalDiscriminator(visual_size+language_size)
        self.ln = nn.LayerNorm([visual_size],elementwise_affine=False)
        
    def forward(self, visual, language):
        
        visual = self.ln(visual)
        visual_fake = torch.cat((visual[1:], visual[0:1]), dim=0)
        Ej = -F.softplus(-self.global_D(visual, language)).mean()
        Em = F.softplus(self.global_D(visual_fake, language)).mean()
        dim = Ej - Em

        return dim

    def temp_constrain(self, feat0, feat1, language):
        
        vis_0 = self.ln(feat0)
        vis_1 = self.ln(feat1)
        visual_fake = torch.cat((vis_0[1:], vis_0[0:1]), dim=0)
        Ej = -F.softplus(-self.global_D(vis_1, language)).mean()
        Em = F.softplus(self.global_D(visual_fake, language)).mean()
        dim = Ej - Em

        return dim

    @ex.capture
    def get_acc(self, visual, unseen_language, label, unseen_label):
        
        bs = visual.shape[0]
        unsee_num = unseen_language.shape[0]
        visual = repeat(visual,'b c -> b u c',u = unsee_num)
        unseen_language = repeat(unseen_language,'u c -> b u c', b = bs)
        dim_list = -F.softplus(-self.global_D(visual, unseen_language)).squeeze(-1)
        _, pred = torch.max(dim_list, 1)
        unseen_label = torch.tensor(unseen_label).cuda()
        pred = torch.index_select(unseen_label,0,pred)
        acc = pred.eq(label.view_as(pred)).float().mean()

        return acc, pred

@ex.capture
def temp_mask(data, mask_frame):
    x = data.clone()
    n, c, t, v, m = x.shape
    remain_num = t - mask_frame
    remain_frame = random.sample(range(t), remain_num)
    remain_frame.sort()
    x = x[:, :, remain_frame, :, :]

    return x
    
def motion_att_temp_mask(data, mask_frame):

    n, c, t, v, m = data.shape
    temp = data.clone()
    remain_num = t - mask_frame

    ## get the motion_attention value
    motion = torch.zeros_like(temp)
    motion[:, :, :-1, :, :] = temp[:, :, 1:, :, :] - temp[:, :, :-1, :, :]
    motion = -(motion)**2
    temporal_att = motion.mean((1,3,4))

    ## The frames with the smallest att are reserved
    _,temp_list = torch.topk(temporal_att, remain_num)
    temp_list,_ = torch.sort(temp_list.squeeze())
    temp_list = repeat(temp_list,'n t -> n c t v m',c=c,v=v,m=m)
    temp_resample = temp.gather(2,temp_list)

    ## random temp mask
    random_frame = random.sample(range(remain_num), remain_num-mask_frame)
    random_frame.sort()
    output = temp_resample[:, :, random_frame, :, :]

    return output
    
def motion_att_temp_mask2(data, mask_frame):

    n, c, t, v, m = data.shape
    temp = data.clone()
    remain_num = t - mask_frame

    ## get the motion_attention value
    motion_pre = torch.zeros_like(temp)
    motion_nex = torch.zeros_like(temp)
    motion_pre[:, :, :-1, :, :] = temp[:, :, 1:, :, :] - temp[:, :, :-1, :, :]
    motion_nex[:, :, 1:, :, :] = temp[:, :, :-1, :, :] - temp[:, :, 1:, :, :]
    motion = -((motion_pre)**2+(motion_nex)**2)
    temporal_att = motion.mean((1,3,4))

    ## The frames with the smallest att are reserved
    _,temp_list = torch.topk(temporal_att, remain_num)
    temp_list,_ = torch.sort(temp_list.squeeze())
    temp_list = repeat(temp_list,'n t -> n c t v m',c=c,v=v,m=m)
    temp_resample = temp.gather(2,temp_list)

    ## random temp mask
    random_frame = random.sample(range(remain_num), remain_num-mask_frame)
    random_frame.sort()
    output = temp_resample[:, :, random_frame, :, :]

    return output
