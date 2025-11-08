import torch
import torch.nn as nn
import torch.nn.functional as F
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss


def total_variation_loss(pred):
    probs = F.softmax(pred, dim=0)  # [C,H,W,D]
    foreground = 1.0 - probs[0]     # [H,W,D]

    dz = torch.abs(foreground[:, :, 1:] - foreground[:, :, :-1]).mean()
    dy = torch.abs(foreground[:, 1:, :] - foreground[:, :-1, :]).mean()
    dx = torch.abs(foreground[1:, :, :] - foreground[:-1, :, :]).mean()

    return dx + dy + dz

def compactness_loss(pred):
    probs = F.softmax(pred, dim=0)
    foreground = 1.0 - probs[0]

    volume = foreground.sum() + 1e-6
    dx = torch.abs(foreground[1:, :, :] - foreground[:-1, :, :]).sum()
    dy = torch.abs(foreground[:, 1:, :] - foreground[:, :-1, :]).sum()
    dz = torch.abs(foreground[:, :, 1:] - foreground[:, :, :-1]).sum()
    surface = dx + dy + dz + 1e-6

    return surface / volume

def soft_lcc_loss(pred, sigma=10.0):
    probs = F.softmax(pred, dim=0)
    foreground = 1.0 - probs[0]

    H, W, D = foreground.shape
    device = foreground.device
    zz, yy, xx = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        torch.arange(D, device=device),
        indexing="ij"
    )
    coords = torch.stack([zz, yy, xx], dim=0).float()

    weights = foreground / (foreground.sum() + 1e-6)
    centroid = (coords * weights).sum(dim=(1,2,3))

    dist2 = ((coords[0] - centroid[0])**2 +
             (coords[1] - centroid[1])**2 +
             (coords[2] - centroid[2])**2)
    mask = torch.exp(-dist2 / (2 * sigma**2))
    mask = mask / (mask.max() + 1e-6)

    soft_lcc = foreground * mask
    return F.mse_loss(foreground, soft_lcc)

def connectedness_regularizer(pred, alpha=0.1, beta=0.1, gamma=0.01):
    tv = total_variation_loss(pred)
    comp = compactness_loss(pred)
    lcc = soft_lcc_loss(pred)
    return alpha * tv + beta * comp + gamma * lcc


class MultiViewConsistencyLoss(nn.Module):
    def __init__(self, base_loss_fn=None, consistency_weight=1.0, consistency_type='dice',
                 conn_alpha=0.1, conn_beta=0.1, conn_gamma=0.01):
        super().__init__()
        self.base_loss_fn = base_loss_fn or DC_and_CE_loss({'batch_dice': True, 'smooth':1e-5, 'do_bg':False}, {})
        self.consistency_weight = consistency_weight
        self.consistency_type = consistency_type
        self.conn_alpha = conn_alpha
        self.conn_beta = conn_beta
        self.conn_gamma = conn_gamma

    def forward(self, predictions, targets=None, case_info=None, consistency_pairs=None):
        loss_dict = {}

        base_loss = self.base_loss_fn(predictions, targets) if targets is not None else torch.tensor(0.0, device=predictions.device)
        loss_dict['base_loss'] = base_loss

        consistency_loss = self._compute_consistency_loss(predictions, case_info)
        loss_dict['consistency_loss'] = consistency_loss

        total_loss = base_loss + self.consistency_weight * consistency_loss
        loss_dict['total_loss'] = total_loss
        return total_loss, loss_dict

    def _compute_consistency_loss(self, predictions, case_info):
        if case_info is None or len(case_info) < 2:
            return torch.tensor(0.0, device=predictions.device)

        patient_groups = {}
        for i, info in enumerate(case_info):
            if isinstance(info, dict) and 'id' in info:
                patient_id = info['id']
                patient_groups.setdefault(patient_id, []).append(i)

        consistency_losses = []
        for patient_id, indices in patient_groups.items():
            if len(indices) >= 2:
                for i in range(len(indices)):
                    for j in range(i+1, len(indices)):
                        idx1, idx2 = indices[i], indices[j]
                        pred1, pred2 = predictions[idx1], predictions[idx2]
                        pair_loss = self._compute_pair_consistency(pred1.unsqueeze(0), pred2.unsqueeze(0))

                        conn_loss1 = connectedness_regularizer(pred1, self.conn_alpha, self.conn_beta, self.conn_gamma)
                        conn_loss2 = connectedness_regularizer(pred2, self.conn_alpha, self.conn_beta, self.conn_gamma)
                        conn_loss = (conn_loss1 + conn_loss2) / 2.0
                        consistency_losses.append(pair_loss + conn_loss)

        if consistency_losses:
            return torch.stack(consistency_losses).mean()
        else:
            return torch.tensor(0.0, device=predictions.device)

    def _compute_pair_consistency(self, pred1, pred2):
        prob1 = F.softmax(pred1, dim=1)
        prob2 = F.softmax(pred2, dim=1)
        if self.consistency_type == 'dice':
            intersection = (prob1 * prob2).sum(dim=(2,3,4))
            union = prob1.sum(dim=(2,3,4)) + prob2.sum(dim=(2,3,4))
            dice = (2.0 * intersection + 1e-5) / (union + 1e-5)
            return 1.0 - dice.mean()
        elif self.consistency_type == 'kl':
            kl1 = F.kl_div(F.log_softmax(pred1, dim=1), prob2, reduction='batchmean')
            kl2 = F.kl_div(F.log_softmax(pred2, dim=1), prob1, reduction='batchmean')
            return 0.5 * (kl1 + kl2)
        elif self.consistency_type == 'mse':
            return F.mse_loss(prob1, prob2)
        else:
            raise ValueError(f"Unknown consistency type: {self.consistency_type}")


def create_multiview_consistency_loss(base_loss_fn=None, consistency_weight=1.0, consistency_type='dice'):
    return MultiViewConsistencyLoss(base_loss_fn, consistency_weight, consistency_type)
