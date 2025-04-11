import torch
import torch.nn as nn
import torch.nn.functional as F

class ADTNetLoss(nn.Module):
    def __init__(self, lambda1=1.0, lambda2=1.0, gamma=0.1):
        super(ADTNetLoss, self).__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.gamma = gamma
        self.ce_loss = nn.BCELoss()
    
    def forward(self, pred_fg, pred_bg, T_fg, T_bg, target):
        """
        Args:
            pred_fg (Tensor): Foreground predictions
            pred_bg (Tensor): Background predictions
            T_fg (Tensor): Foreground threshold map
            T_bg (Tensor): Background threshold map
            target (Tensor): Ground truth binary mask
        """
        # Compute foreground and background losses
        loss_fg = self.ce_loss(pred_fg, target)
        loss_bg = self.ce_loss(pred_bg, 1 - target)
        
        # Compute threshold divergence loss
        loss_thresh = torch.mean((T_fg - T_bg) ** 2)
        
        # Combine losses
        total_loss = (self.lambda1 * loss_fg + 
                     self.lambda2 * loss_bg + 
                     self.gamma * loss_thresh)
        
        return total_loss