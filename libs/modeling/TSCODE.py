import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import Scale,LayerNorm


class DeCLS(nn.Module):
    def __init__(self,n_embd,level=6):
        super().__init__()
        self.level = level
        self.contrans=nn.ConvTranspose1d(n_embd,n_embd,kernel_size=3,stride=2,padding=1,output_padding=1)
        self.conv_l = nn.Conv1d(n_embd, n_embd, kernel_size=1, stride=1, padding=0)
        self.attn=nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(n_embd, n_embd, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self,fpn_feates):
        cls_feates=[]
        for idx in range(self.level):
            if idx < self.level-1:
                l_high=self.contrans(fpn_feates[idx+1])
                l=self.conv_l(fpn_feates[idx])
                l2=F.relu(l_high+l)
                x_sce=self.attn(l2)*l_high+fpn_feates[idx]
                cls_feates.append(x_sce)
        cls_feates.append(fpn_feates[-1])
        return cls_feates


class DeReg(nn.Module):
    def __init__(self,n_embd,level=6):
        super().__init__()
        self.level=level
        self.down=nn.Conv1d(n_embd,n_embd,kernel_size=3,stride=2,padding=1)
        self.conv_l=nn.Conv1d(n_embd,n_embd,kernel_size=1,stride=1,padding=0)
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(n_embd, n_embd, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self,fpn_feates):
        loc_feates=[]
        loc_feates.append(fpn_feates[0])
        for idx in range(self.level):
            if idx < self.level-1:
                l_low=self.down(fpn_feates[idx])
                l=self.conv_l(fpn_feates[idx+1])
                l2=F.relu(l_low+l)
                x_reg=self.attn(l2)*l_low+fpn_feates[idx+1]
                loc_feates.append(x_reg)
        return loc_feates


class CrossLevel(nn.Module):
    def __init__(self, n_embd, level=6) -> None:
        super(CrossLevel,self).__init__()
        self.level = level
        self.conv = nn.Conv1d(n_embd, n_embd, kernel_size=3, stride=2, padding=1)
        self.convtrans = nn.ConvTranspose1d(n_embd, n_embd, kernel_size=3, stride=2, padding=1,
                                            output_padding=1)
        # self.change_channel =nn.Conv1d(n_embd*2,n_embd*3,kernel_size=1)

    def forward(self, fpn_feates):
        new_fpn_feates=[]
        for i, idx in enumerate(range(self.level)):
            if i==0:
                x1 = self.convtrans(fpn_feates[i + 1])
                x2=fpn_feates[i]
                # new_fpn_feates.append(self.change_channel(torch.cat([x1, x2], dim=1)))
                new_fpn_feates.append(x1+x2)
            elif self.level-1>i>0:
                x1=self.convtrans(fpn_feates[i+1])
                x2=fpn_feates[i]
                x3=self.conv(fpn_feates[i-1])
                # new_fpn_feates.append(torch.cat([x1,x2,x3],dim=1))
                new_fpn_feates.append(x1+x2+x3)
            else:
                x2=fpn_feates[i]
                x3=self.conv(fpn_feates[i-1])
                # new_fpn_feates.append(self.change_channel(torch.cat([x2,x3],dim=1)))
                new_fpn_feates.append(x2+x3)
        return new_fpn_feates


class Refine(nn.Module):
    def __init__(self,n_embed,fpn_levels=6):
        super().__init__()
        self.reg_offset=nn.Sequential(
            nn.Conv1d(n_embed,n_embed//4,kernel_size=1),
            LayerNorm(n_embed//4),
            nn.ReLU(),
            nn.Conv1d(n_embed//4,2,kernel_size=3,padding=1)
        )
        self.scale = nn.ModuleList()
        for idx in range(fpn_levels):
            self.scale.append(Scale())
        self.norm=LayerNorm(2)

    def forward(self,fpn_feates):
        cls_prob,reg_offset=[],[]
        for l,feat in enumerate(fpn_feates):
            # cls_prob.append(self.cls(feat))
            reg_offset.append(F.relu(self.scale[l](self.norm(self.reg_offset(feat)))))
        return cls_prob,reg_offset

