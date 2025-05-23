import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from structure_aware_diffusion_policy.model.vision.pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction, PointNetFeaturePropagation

class PointNet2(nn.Module):
    
    def __init__(self, normal_channel=False, feature_dim=128):
        super(PointNet2, self).__init__()
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=3 + 3 + additional_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fp3 = PointNetFeaturePropagation(in_channel=1280, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128+6+additional_channel, mlp=[128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, feature_dim, 1)  # 输出 feature_dim 维特征向量

    def forward(self, xyz):
        # Set Abstraction layers
        if xyz.shape[1] != 3:
            xyz = xyz.permute(0, 2, 1)
        B,C,N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:,:3,:]
        else:
            l0_points = xyz
            l0_xyz = xyz
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([l0_xyz,l0_points],1), l1_points)
        # FC layers
        feat = F.relu(self.bn1(self.conv1(l0_points)))
        output = self.drop1(feat)
        output = self.conv2(output)
        output = output.permute(0, 2, 1)  # 输出形状 (B, N, feature_dim)
        return output


class PointNet2Global(nn.Module):
    def __init__(self, affordance_feature=False, feature_dim=32):
        super(PointNet2Global, self).__init__()
        if affordance_feature:
            additional_channel = 2
        else:
            additional_channel = 0
        self.affordance_feature = affordance_feature
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=3 + 3 + additional_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 256, 512], group_all=True)
        self.fc = nn.Sequential(
            nn.Linear(512, feature_dim),
            nn.ReLU(),
            nn.BatchNorm1d(feature_dim)
        )
    def forward(self, xyz):
        # Set Abstraction layers
        if self.affordance_feature:
            if xyz.shape[1] != 5:
                xyz = xyz.permute(0, 2, 1)
        else:
            if xyz.shape[1] != 3:
                xyz = xyz.permute(0, 2, 1)
                
        B,C,N = xyz.shape
        
        if self.affordance_feature:
            l0_points = xyz
            l0_xyz = xyz[:,:3,:]
        else:
            l0_points = xyz
            l0_xyz = xyz
       
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # max_pooling
        global_feat = l3_points.squeeze(-1)  # (B, 512)
        # fully connected layers, downsample to feature_dim
        global_feat = self.fc(global_feat)
        return global_feat