import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmdet.registry import MODELS
from mmcv.cnn.bricks.transformer import MultiheadAttention
from mmcv.cnn import build_norm_layer
from torch.nn.init import normal_

@MODELS.register_module()
class UncertaintyFusionModule(BaseModule):
    def __init__(self,
                 in_channels,
                 fusion_channels=256,
                 num_levels=4,
                 num_heads=8,
                 dropout=0.1,
                 norm_cfg=dict(type='LN'),
                 lambda1=0.1,
                 lambda2=0.1):
        super().__init__()
        self.num_levels = num_levels
        self.fusion_channels = fusion_channels
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        # 可学习的权重
        self.weight_entropy = nn.Parameter(torch.tensor(0.5))
        self.weight_distance = nn.Parameter(torch.tensor(0.5))

        # 交叉注意力模块
        self.cross_attns = nn.ModuleList([
            MultiheadAttention(
                embed_dims=fusion_channels,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_levels)
        ])

        # 投影层
        self.img_proj = nn.Linear(in_channels, fusion_channels)
        self.seg_proj = nn.Linear(in_channels, fusion_channels)
        self.entropy_proj = nn.Linear(in_channels, fusion_channels)
        self.distance_proj = nn.Linear(in_channels, fusion_channels)

        # 位置编码
        self.level_pos_embed = nn.Parameter(
            torch.zeros(num_levels, 1, fusion_channels))

        # Layer Normalization
        self.norms = nn.ModuleList([
            build_norm_layer(norm_cfg, fusion_channels)[1]
            for _ in range(num_levels)
        ])

        # 初始化
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        normal_(self.level_pos_embed, std=0.02)

    def forward(self, img_feats, seg_feats, entropy_feats, distance_feats, encoder):
        """
        Args:
            img_feats (list[Tensor]): 多尺度图像特征 
            seg_feats (list[Tensor]): 语义分割特征
            entropy_feats (list[Tensor]): softmax entropy 特征
            distance_feats (list[Tensor]): softmax distance 特征
            encoder: GroundingDinoTransformerEncoder实例
        """
        enhanced_feats = []
        curr_feat = img_feats[0]

        for i in range(self.num_levels):
            # 1. 图像和分割特征拼接并投影
            img_seg_feat = torch.cat((img_feats[i], seg_feats[i]), dim=-1)
            img_seg_proj = self.img_proj(img_seg_feat)

            # 2. 加权融合 entropy 特征
            entropy_proj = self.entropy_proj(entropy_feats[i])
            fused_feat = img_seg_proj + self.weight_entropy * entropy_proj

            # 3. 交叉融合 distance 特征
            distance_proj = self.distance_proj(distance_feats[i])
            fused_feat = self.cross_attns[i](
                query=fused_feat,
                key=distance_proj,
                value=distance_proj,
                key_pos=self.get_position_encoding(distance_proj, i),
                need_weights=False
            )[0]

            # 4. 编码器增强
            enhanced_feat = encoder.forward_single_level(fused_feat, i)

            # 5. 保存结果并更新当前特征
            enhanced_feats.append(enhanced_feat)
            if i < self.num_levels - 1:
                curr_feat = enhanced_feat

        return enhanced_feats

    def compute_loss(self, features):
        """计算正则化损失"""
        loss = 0
        for i in range(len(features) - 1):
            fi = features[i]
            fi1 = features[i + 1]
            loss += self.lambda1 * torch.sum(torch.abs(fi * fi1))
            loss += self.lambda2 * torch.sum((fi * fi1) ** 2)
        return loss

    def get_position_encoding(self, x, level_idx):
        """生成位置编码
        Args:
            x (Tensor): 输入特征 [B, N, C]
            level_idx (int): 特征层级索引
        """
        B, N, C = x.shape
        pos_embed = self.level_pos_embed[level_idx]  # [1, C]
        pos_embed = pos_embed.expand(B, N, -1)  # [B, N, C]
        return pos_embed