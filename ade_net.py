import torch
import torch.nn as nn
import torch.nn.functional as F

class ADTNet(nn.Module):
    def __init__(self, in_channels=1):
        super(ADTNet, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(5.0))  # Foreground sigmoid steepness
        self.beta = nn.Parameter(torch.tensor(3.0))   # Background sigmoid steepness
        
        # Threshold predictors
        self.foreground_predictor = ThresholdPredictor(in_channels, local_context=True)
        self.background_predictor = ThresholdPredictor(in_channels, local_context=False)
        
    def forward(self, anomaly_scores, detection_mask):
        """
        Args:
            anomaly_scores (Tensor): Raw anomaly scores [B, 1, H, W]
            detection_mask (Tensor): Binary detection mask [B, 1, H, W]
        """
        # Create foreground and background masks
        fg_mask = detection_mask
        bg_mask = 1 - detection_mask
        
        # Compute region-wise means
        fg_mean = (anomaly_scores * fg_mask).sum(dim=(2,3)) / (fg_mask.sum(dim=(2,3)) + 1e-6)
        bg_mean = (anomaly_scores * bg_mask).sum(dim=(2,3)) / (bg_mask.sum(dim=(2,3)) + 1e-6)
        
        fg_mean = fg_mean.view(-1, 1, 1, 1).expand_as(anomaly_scores)
        bg_mean = bg_mean.view(-1, 1, 1, 1).expand_as(anomaly_scores)
        
        # Adaptive Region Normalization
        norm_scores = torch.where(
            fg_mask > 0,
            0.5 / (1 + torch.exp(-self.alpha * (anomaly_scores - fg_mean))) + 0.3,
            0.5 / (1 + torch.exp(-self.beta * (anomaly_scores - bg_mean))) + 0.3
        )
        
        # Predict thresholds
        T_fg = self.foreground_predictor(norm_scores * fg_mask)
        T_bg = self.background_predictor(norm_scores * bg_mask)
        
        # Differentiable binarization
        delta = 0.1
        prob_fg = self._compute_probability(norm_scores, T_fg, delta, fg_mask)
        prob_bg = self._compute_probability(norm_scores, T_bg, delta, bg_mask)
        
        # Combine probabilities
        final_prob = torch.max(prob_fg, prob_bg)
        
        return final_prob, T_fg, T_bg, norm_scores
    
    def _compute_probability(self, scores, threshold, delta, mask):
        condition1 = (scores >= threshold + delta)
        condition2 = (scores >= threshold) & (scores < threshold + delta)
        
        prob = torch.zeros_like(scores)
        prob[condition1] = 1.0
        prob[condition2] = (scores[condition2] - threshold[condition2] + delta) / delta
        prob = prob * mask
        
        return prob

class ThresholdPredictor(nn.Module):
    def __init__(self, in_channels, local_context=True):
        super(ThresholdPredictor, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, 1, kernel_size=1)
        
        self.local_context = local_context
        if local_context:
            self.context_conv = nn.Conv2d(64, 64, kernel_size=3, padding=1, dilation=2)
        else:
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            self.context_fc = nn.Linear(64, 64)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Apply context modeling
        if self.local_context:
            context = self.context_conv(x)
            x = x + context
        else:
            context = self.global_pool(x)
            context = self.context_fc(context.squeeze(-1).squeeze(-1))
            context = context.view(x.size(0), -1, 1, 1)
            x = x * context
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.sigmoid(self.conv4(x))
        
        return x