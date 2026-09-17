import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CostMatrixLoss(nn.Module):
    """
    Risk-sensitive expected cost loss given a semantic / ordinal cost matrix C in R^{K x K}.
    L_cost = sum_{j=0}^{K-1} p_j * C_{y, j}
    where p_j is the predicted probability for class j, and y is ground truth index.
    """
    def __init__(self, cost_matrix=None):
        super().__init__()
        if cost_matrix is not None:
            if isinstance(cost_matrix, np.ndarray):
                cost_matrix = torch.from_numpy(cost_matrix).float()
            elif not isinstance(cost_matrix, torch.Tensor):
                cost_matrix = torch.tensor(cost_matrix, dtype=torch.float32)
            self.register_buffer('cost_matrix', cost_matrix)
        else:
            self.cost_matrix = None

    def forward(self, logits, targets):
        """
        logits: [B, K]
        targets: [B]
        """
        if self.cost_matrix is None:
            # Default ordinal 1D distance |i - j|
            num_classes = logits.shape[-1]
            idx = torch.arange(num_classes, device=logits.device, dtype=torch.float32)
            cost_mat = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))
        else:
            cost_mat = self.cost_matrix.to(logits.device)
            
        probs = F.softmax(logits, dim=-1)  # [B, K]
        batch_costs = cost_mat[targets]     # [B, K]
        loss = (probs * batch_costs).sum(dim=-1).mean()
        return loss


class SoftTargetKLLoss(nn.Module):
    """
    Soft-target KL divergence loss using semantic cost matrix as temperature-smoothed ground cost.
    q_j(y) = exp(-C_{y, j} / tau) / sum_k exp(-C_{y, k} / tau)
    L_KL = KL(q(y) || p)
    """
    def __init__(self, cost_matrix=None, tau_smooth=1.0):
        super().__init__()
        if cost_matrix is not None:
            if isinstance(cost_matrix, np.ndarray):
                cost_matrix = torch.from_numpy(cost_matrix).float()
            elif not isinstance(cost_matrix, torch.Tensor):
                cost_matrix = torch.tensor(cost_matrix, dtype=torch.float32)
            self.register_buffer('cost_matrix', cost_matrix)
        else:
            self.cost_matrix = None
        self.tau_smooth = tau_smooth

    def forward(self, logits, targets):
        if self.cost_matrix is None:
            num_classes = logits.shape[-1]
            idx = torch.arange(num_classes, device=logits.device, dtype=torch.float32)
            cost_mat = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))
        else:
            cost_mat = self.cost_matrix.to(logits.device)

        batch_costs = cost_mat[targets]  # [B, K]
        soft_targets = F.softmax(-batch_costs / self.tau_smooth, dim=-1)  # [B, K]
        log_probs = F.log_softmax(logits, dim=-1)
        loss = F.kl_div(log_probs, soft_targets, reduction='batchmean')
        return loss


class SymmetricContrastiveLoss(nn.Module):
    """
    Symmetric InfoNCE image-to-text and text-to-image contrastive loss.
    """
    def __init__(self):
        super().__init__()

    def forward(self, image_features, text_features, targets, logit_scale):
        """
        image_features: [B, D] normalized
        text_features: [K, D] normalized
        targets: [B]
        logit_scale: scalar or tensor
        """
        if isinstance(logit_scale, torch.Tensor):
            scale = logit_scale.exp() if logit_scale.numel() == 1 else 100.0
        else:
            scale = float(logit_scale)

        # Image to text
        logits_i2t = scale * (image_features @ text_features.t())  # [B, K]
        loss_i2t = F.cross_entropy(logits_i2t, targets)

        # Text to image (handling multi-instance classes in batch)
        logits_t2i = logits_i2t.t()  # [K, B]
        K = text_features.size(0)
        target_mask = (targets.unsqueeze(0) == torch.arange(K, device=targets.device).unsqueeze(1)).float()
        row_sums = target_mask.sum(dim=1, keepdim=True)
        valid_classes = (row_sums.squeeze(1) > 0)

        if valid_classes.any():
            target_dist = target_mask[valid_classes] / row_sums[valid_classes]
            log_probs_t2i = F.log_softmax(logits_t2i[valid_classes], dim=-1)
            loss_t2i = F.kl_div(log_probs_t2i, target_dist, reduction='batchmean')
            return 0.5 * (loss_i2t + loss_t2i), logits_i2t
        
        return loss_i2t, logits_i2t


class PromptSRCLoss(nn.Module):
    """
    PromptSRC Regularization Loss:
    1. Self-Consistency Loss (SCL): KL divergence between zero-shot frozen predictions and tuned predictions.
    2. Text feature cosine regularization: 1 - cos(t_tuned, t_zs).
    3. Vision feature cosine regularization: 1 - cos(f_tuned, f_zs).
    """
    def __init__(self, temperature=2.0, weight_scl=1.0, weight_text=1.0, weight_vis=1.0):
        super().__init__()
        self.temperature = temperature
        self.weight_scl = weight_scl
        self.weight_text = weight_text
        self.weight_vis = weight_vis

    def forward(self, tuned_logits, zs_logits, tuned_text_feat, zs_text_feat, tuned_vis_feat=None, zs_vis_feat=None):
        # 1. SCL
        p_zs = F.softmax(zs_logits / self.temperature, dim=-1)
        log_p_tuned = F.log_softmax(tuned_logits / self.temperature, dim=-1)
        loss_scl = (self.temperature ** 2) * F.kl_div(log_p_tuned, p_zs, reduction='batchmean')

        # 2. Text Feature Regularization
        cos_sim_text = F.cosine_similarity(tuned_text_feat, zs_text_feat, dim=-1)
        loss_text = (1.0 - cos_sim_text).mean()

        # 3. Vision Feature Regularization
        loss_vis = torch.tensor(0.0, device=tuned_logits.device)
        if tuned_vis_feat is not None and zs_vis_feat is not None:
            cos_sim_vis = F.cosine_similarity(tuned_vis_feat, zs_vis_feat, dim=-1)
            loss_vis = (1.0 - cos_sim_vis).mean()

        total_loss = (self.weight_scl * loss_scl + 
                      self.weight_text * loss_text + 
                      self.weight_vis * loss_vis)
        
        metrics = {
            "scl": loss_scl.item(),
            "text_reg": loss_text.item(),
            "vis_reg": loss_vis.item()
        }
        return total_loss, metrics


class CompositeCriterion(nn.Module):
    """
    Composite loss criterion combining base classification, optional ordinal cost, and PromptSRC regularizer.
    """
    def __init__(self, 
                 base_loss_type='ce', 
                 use_ordinal=False, 
                 ordinal_type='expected_cost',
                 cost_matrix=None, 
                 lambda_ord=1.0, 
                 use_promptsrc=False, 
                 lambda_src=1.0, 
                 src_temp=2.0,
                 use_kgcoop=False,
                 lambda_kg=2.0):
        super().__init__()
        self.base_loss_type = base_loss_type
        self.use_ordinal = use_ordinal
        self.lambda_ord = lambda_ord
        self.use_promptsrc = use_promptsrc
        self.lambda_src = lambda_src
        self.use_kgcoop = use_kgcoop
        self.lambda_kg = lambda_kg

        if base_loss_type == 'contrastive':
            self.base_criterion = SymmetricContrastiveLoss()
        else:
            self.base_criterion = nn.CrossEntropyLoss()

        if use_ordinal:
            if ordinal_type == 'soft_kl':
                self.ord_criterion = SoftTargetKLLoss(cost_matrix=cost_matrix)
            else:
                self.ord_criterion = CostMatrixLoss(cost_matrix=cost_matrix)
        else:
            self.ord_criterion = None

        if use_promptsrc:
            self.promptsrc_criterion = PromptSRCLoss(temperature=src_temp)
        else:
            self.promptsrc_criterion = None

    def forward(self, 
                logits, 
                targets, 
                image_features=None, 
                text_features=None, 
                logit_scale=100.0,
                zs_logits=None, 
                zs_text_features=None, 
                zs_image_features=None):
        
        loss_dict = {}

        # 1. Base loss
        if self.base_loss_type == 'contrastive' and image_features is not None and text_features is not None:
            base_loss, _ = self.base_criterion(image_features, text_features, targets, logit_scale)
        else:
            base_loss = self.base_criterion(logits, targets)
        loss_dict['base_loss'] = base_loss.item()
        total_loss = base_loss

        # 2. Ordinal loss
        if self.use_ordinal and self.ord_criterion is not None:
            ord_loss = self.ord_criterion(logits, targets)
            loss_dict['ord_loss'] = ord_loss.item()
            total_loss = total_loss + self.lambda_ord * ord_loss

        # 3. PromptSRC loss
        if self.use_promptsrc and self.promptsrc_criterion is not None and zs_logits is not None:
            src_loss, src_metrics = self.promptsrc_criterion(
                tuned_logits=logits,
                zs_logits=zs_logits,
                tuned_text_feat=text_features,
                zs_text_feat=zs_text_features,
                tuned_vis_feat=image_features,
                zs_vis_feat=zs_image_features
            )
            loss_dict['src_loss'] = src_loss.item()
            loss_dict.update(src_metrics)
            total_loss = total_loss + self.lambda_src * src_loss

        # 4. kgCoOp loss: penalizes deviation from zero-shot base anchor
        if self.use_kgcoop and text_features is not None and zs_text_features is not None:
            cos_sim_kg = F.cosine_similarity(text_features, zs_text_features, dim=-1)
            kg_loss = (1.0 - cos_sim_kg).mean()
            loss_dict['kg_loss'] = kg_loss.item()
            total_loss = total_loss + self.lambda_kg * kg_loss

        loss_dict['total_loss'] = total_loss.item()
        return total_loss, loss_dict

