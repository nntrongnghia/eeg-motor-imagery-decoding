import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

def softmax_focal_loss(pred_logits, targets, gamma=2, alpha=0.25):
    p = torch.softmax(pred_logits, -1)
    targets = F.one_hot(targets.to(torch.int64), num_classes=pred_logits.shape[-1]).to(torch.float)
    ce_loss = F.binary_cross_entropy_with_logits(
        pred_logits, targets, reduction="none"
    )
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean()*10