import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Multi-class Dice Loss, input logits [B, C, D, H, W], target integer labels [B, D, H, W]"""
    def __init__(self, num_classes: int, smooth: float = 1e-6, ignore_background: bool = False):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_background = ignore_background

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        target_one_hot = F.one_hot(target.long(), num_classes=self.num_classes).permute(0, 4, 1, 2, 3).float()
        dims = (0, 2, 3, 4)
        intersection = (probs * target_one_hot).sum(dims)
        cardinality = (probs + target_one_hot).sum(dims)
        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        if self.ignore_background and self.num_classes > 1:
            dice_per_class = dice_per_class[1:]
        loss = 1.0 - dice_per_class.mean()
        return loss


class FocalLoss(nn.Module):
    """Focal Loss"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CombinedLoss(nn.Module):
    """Combined Loss: Dice + CrossEntropy"""
    def __init__(self, num_classes: int, dice_weight=0.5, ce_weight=0.5, class_weights=None,
                 ignore_background: bool = False):
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.class_weights = class_weights
        self.dice_loss = DiceLoss(num_classes=num_classes, ignore_background=ignore_background)
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        if self.class_weights is not None:
            device = inputs.device
            class_weights = self.class_weights.to(device)
            ce_loss = F.cross_entropy(inputs, targets, weight=class_weights)
        else:
            ce_loss = F.cross_entropy(inputs, targets)

        dice_loss = self.dice_loss(inputs, targets)
        total_loss = self.dice_weight * dice_loss + self.ce_weight * ce_loss
        return total_loss


class ContrastiveLoss(nn.Module):
    """Contrastive Learning Loss - InfoNCE Loss"""
    def __init__(self, temperature=0.07, negative_mode='unpaired'):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.negative_mode = negative_mode

    def forward(self, features, labels=None):
        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        positive_mask = torch.eye(features.size(0), device=features.device)
        exp_sim = torch.exp(similarity_matrix)
        log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
        mean_log_prob = (positive_mask * log_prob).sum(dim=1) / positive_mask.sum(dim=1)
        return -mean_log_prob.mean()


class MultiViewContrastiveLoss(nn.Module):
    """Multi-View Contrastive Learning Loss"""
    def __init__(self, temperature=0.07, lambda_intra=1.0, lambda_inter=1.0):
        super(MultiViewContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.lambda_intra = lambda_intra
        self.lambda_inter = lambda_inter

    def forward(self, sag_features, cor_features, tra_features, patient_ids):
        batch_size = sag_features.size(0)
        device = sag_features.device

        sag_features = F.normalize(sag_features, dim=1)
        cor_features = F.normalize(cor_features, dim=1)
        tra_features = F.normalize(tra_features, dim=1)

        intra_loss = self._compute_intra_patient_loss(sag_features, cor_features, tra_features)
        inter_loss = self._compute_inter_patient_loss(sag_features, cor_features, tra_features, patient_ids)

        total_loss = self.lambda_intra * intra_loss + self.lambda_inter * inter_loss
        return total_loss, intra_loss, inter_loss

    def _compute_intra_patient_loss(self, sag_features, cor_features, tra_features):
        positive_pairs = [
            (sag_features, cor_features),
            (sag_features, tra_features),
            (cor_features, tra_features)
        ]

        intra_loss = 0.0
        for feat1, feat2 in positive_pairs:
            sim = torch.sum(feat1 * feat2, dim=1) / self.temperature
            intra_loss += -torch.mean(sim)

        return intra_loss / len(positive_pairs)

    def _compute_inter_patient_loss(self, sag_features, cor_features, tra_features, patient_ids):
        batch_size = sag_features.size(0)
        patient_ids = patient_ids.view(-1, 1)
        same_patient_mask = (patient_ids == patient_ids.T).float()

        all_features = torch.cat([sag_features, cor_features, tra_features], dim=0)
        similarity_matrix = torch.matmul(all_features, all_features.T) / self.temperature

        negative_mask = 1 - same_patient_mask
        negative_mask = negative_mask.repeat(3, 3)

        exp_sim = torch.exp(similarity_matrix)
        log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
        negative_loss = (negative_mask * log_prob).sum(dim=1) / (negative_mask.sum(dim=1) + 1e-8)

        return -torch.mean(negative_loss)


class HardNegativeMiningLoss(nn.Module):
    """Contrastive Loss with Hard Negative Mining"""
    def __init__(self, temperature=0.07, top_k=10):
        super(HardNegativeMiningLoss, self).__init__()
        self.temperature = temperature
        self.top_k = top_k

    def forward(self, features, labels):
        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        labels = labels.view(-1, 1)
        label_mask = (labels == labels.T).float()

        batch_size = features.size(0)
        hard_negative_loss = 0.0

        for i in range(batch_size):
            neg_sim = similarity_matrix[i] * (1 - label_mask[i])
            if neg_sim.sum() > 0:
                top_k_neg_sim, _ = torch.topk(neg_sim, min(self.top_k, (neg_sim > 0).sum()))
                hard_negative_loss += torch.mean(top_k_neg_sim)

        return hard_negative_loss / batch_size


class MultiScaleContrastiveLoss(nn.Module):
    """Multi-Scale Contrastive Learning Loss"""
    def __init__(self, temperature=0.07, scale_weights=None):
        super(MultiScaleContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.scale_weights = scale_weights or [1.0, 0.5, 0.25]

    def forward(self, multi_scale_features):
        total_loss = 0.0
        for i, features in enumerate(multi_scale_features):
            if i < len(self.scale_weights):
                weight = self.scale_weights[i]
                scale_loss = ContrastiveLoss(temperature=self.temperature)(features)
                total_loss += weight * scale_loss
        return total_loss


def get_loss_function(config):
    loss_type = config['loss']['type']
    if loss_type == 'dice':
        return DiceLoss(num_classes=config['model']['n_classes'])
    elif loss_type == 'focal':
        return FocalLoss(
            alpha=config['loss']['focal_alpha'],
            gamma=config['loss']['focal_gamma']
        )
    elif loss_type == 'combined':
        class_weights = None
        if 'class_weights' in config['loss']:
            class_weights = torch.tensor(config['loss']['class_weights'], dtype=torch.float32)
        return CombinedLoss(
            num_classes=config['model']['n_classes'],
            dice_weight=config['loss']['dice_weight'],
            ce_weight=config['loss']['ce_weight'],
            class_weights=class_weights,
            ignore_background=False
        )
    elif loss_type == 'contrastive':
        return ContrastiveLoss(temperature=config['loss'].get('temperature', 0.07))
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def get_contrastive_loss_function(config):
    loss_config = config.get('contrastive_loss', {})
    if loss_config.get('type') == 'multi_view':
        return MultiViewContrastiveLoss(
            temperature=loss_config.get('temperature', 0.07),
            lambda_intra=loss_config.get('lambda_intra', 1.0),
            lambda_inter=loss_config.get('lambda_inter', 1.0)
        )
    elif loss_config.get('type') == 'hard_negative':
        return HardNegativeMiningLoss(
            temperature=loss_config.get('temperature', 0.07),
            top_k=loss_config.get('top_k', 10)
        )
    elif loss_config.get('type') == 'multi_scale':
        return MultiScaleContrastiveLoss(
            temperature=loss_config.get('temperature', 0.07),
            scale_weights=loss_config.get('scale_weights', [1.0, 0.5, 0.25])
        )
    else:
        return ContrastiveLoss(temperature=loss_config.get('temperature', 0.07))