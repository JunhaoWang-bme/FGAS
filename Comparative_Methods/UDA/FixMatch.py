import torch
import torch.nn as nn
import torch.nn.functional as F
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.training.loss_functions.deep_supervision import DeepSupervisionWrapper
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.abstract_transforms import Compose
import numpy as np


class FixMatchTrainer:
    def __init__(self, model, num_classes, device, threshold=0.95, lambda_unsup=1.0):
        self.model = model.to(device)
        self.num_classes = num_classes
        self.device = device
        self.threshold = threshold  # Confidence threshold for pseudo-labels
        self.lambda_unsup = lambda_unsup  # Weight for unsupervised loss

        # nnUNet's default Dice + CE loss
        self.supervised_loss = DC_and_CE_loss(
            soft_dice_kwargs={'batch_dice': True, 'smooth': 1e-5, 'do_bg': False},
            ce_kwargs={'ignore_index': -1},
            weight_ce=1, weight_dice=1
        )

    def get_transforms(self):
        # Weak augmentation: mild spatial transforms
        weak_transform = Compose([
            MirrorTransform(axes=(0, 1, 2)),
            SpatialTransform(
                patch_size=(128, 128, 64),
                patch_center_dist_from_border=(32, 32, 16),
                do_elastic_deform=False,
                alpha=(0, 0),
                sigma=(0, 0),
                do_rotation=True,
                angle_x=(-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
                angle_y=(-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
                angle_z=(-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
                do_scale=True,
                scale=(0.9, 1.1),
                border_mode_data='constant',
                border_cval_data=0,
                border_mode_seg='constant',
                border_cval_seg=0,
                order_data=3,
                order_seg=0,
                random_crop=True,
                p_el_per_sample=0,
                p_rot_per_sample=0.2,
                p_scale_per_sample=0.2
            )
        ])

        # Strong augmentation: elastic deform, noise, gamma transforms
        strong_transform = Compose([
            MirrorTransform(axes=(0, 1, 2)),
            SpatialTransform(
                patch_size=(128, 128, 64),
                patch_center_dist_from_border=(32, 32, 16),
                do_elastic_deform=True,
                alpha=(100., 300.),
                sigma=(10., 13.),
                do_rotation=True,
                angle_x=(-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
                angle_y=(-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
                angle_z=(-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
                do_scale=True,
                scale=(0.8, 1.2),
                border_mode_data='constant',
                border_cval_data=0,
                border_mode_seg='constant',
                border_cval_seg=0,
                order_data=3,
                order_seg=0,
                random_crop=True,
                p_el_per_sample=0.2,
                p_rot_per_sample=0.3,
                p_scale_per_sample=0.3
            ),
            GaussianNoiseTransform(p_per_sample=0.1),
            GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2),
            BrightnessMultiplicativeTransform(multiplier_range=(0.7, 1.5), p_per_sample=0.3),
            GammaTransform(gamma_range=(0.7, 1.5), invert_image=False, per_channel=True, p_per_sample=0.3),
            GammaTransform(gamma_range=(0.7, 1.5), invert_image=True, per_channel=True, p_per_sample=0.3)
        ])

        return weak_transform, strong_transform

    def generate_pseudo_labels(self, unlabeled_data):
        # Generate high-confidence pseudo-labels with weak augmentation
        self.model.eval()
        with torch.no_grad():
            weak_transform, _ = self.get_transforms()
            data_dict = {'data': unlabeled_data, 'seg': np.zeros_like(unlabeled_data)}
            augmented = weak_transform(**data_dict)
            weak_data = torch.from_numpy(augmented['data']).to(self.device).float()

            outputs = self.model(weak_data)
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # Main output from deep supervision

            probs = F.softmax(outputs, dim=1)
            max_probs, pseudo_labels = torch.max(probs, dim=1)
            mask = (max_probs >= self.threshold).float()  # Confidence mask

            return pseudo_labels, mask, weak_data.shape

    def consistency_loss(self, strong_data, pseudo_labels, mask):
        # Calculate consistency loss for high-confidence regions
        outputs = self.model(strong_data)
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        loss = 0
        for c in range(1, self.num_classes):  # Skip background class
            pred_c = F.softmax(outputs, dim=1)[:, c]
            target_c = (pseudo_labels == c).float()
            masked_pred = pred_c * mask
            masked_target = target_c * mask

            intersection = (masked_pred * masked_target).sum()
            union = masked_pred.sum() + masked_target.sum() + 1e-5
            dice = 2 * intersection / union
            loss += (1 - dice)

        ce_loss = F.cross_entropy(outputs, pseudo_labels, reduction='none')
        masked_ce = (ce_loss * mask).mean()
        loss += masked_ce

        return loss / (self.num_classes - 1)

    def train_step(self, labeled_batch, unlabeled_batch, optimizer):
        # Single training step with FixMatch
        self.model.train()

        # Supervised loss calculation
        labeled_data = to_cuda(maybe_to_torch(labeled_batch['data']), self.device).float()
        labeled_seg = to_cuda(maybe_to_torch(labeled_batch['seg']), self.device).long()

        sup_outputs = self.model(labeled_data)
        loss_sup = self.supervised_loss(sup_outputs, labeled_seg)

        # Unsupervised FixMatch loss
        if unlabeled_batch is not None:
            pseudo_labels, mask, data_shape = self.generate_pseudo_labels(unlabeled_batch['data'])

            # Strong augmentation for consistency
            _, strong_transform = self.get_transforms()
            data_dict = {'data': unlabeled_batch['data'], 'seg': np.zeros_like(unlabeled_batch['data'])}
            augmented = strong_transform(**data_dict)
            strong_data = torch.from_numpy(augmented['data']).to(self.device).float()

            # Resize mask and pseudo-labels if needed
            if mask.shape != strong_data.shape[2:]:
                mask = F.interpolate(
                    mask.unsqueeze(1).float(),
                    size=strong_data.shape[2:],
                    mode='nearest'
                ).squeeze(1)
                pseudo_labels = F.interpolate(
                    pseudo_labels.unsqueeze(1).float(),
                    size=strong_data.shape[2:],
                    mode='nearest'
                ).squeeze(1).long()

            loss_unsup = self.consistency_loss(strong_data, pseudo_labels, mask)
            total_loss = loss_sup + self.lambda_unsup * loss_unsup
        else:
            total_loss = loss_sup
            loss_unsup = torch.tensor(0.0)

        # Optimization step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        return {
            'total_loss': total_loss.item(),
            'supervised_loss': loss_sup.item(),
            'unsupervised_loss': loss_unsup.item() if unlabeled_batch is not None else 0.0,
            'confidence_mask_ratio': mask.mean().item() if unlabeled_batch is not None else 0.0
        }


if __name__ == "__main__":
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # nnUNet model initialization
    num_classes = 4
    base_num_features = 32
    num_pool = 4
    net_num_pool_op_kernel_sizes = [(2, 2, 2)] * num_pool
    net_conv_kernel_sizes = [(3, 3, 3)] * (num_pool + 1)

    nnunet_model = Generic_UNet(
        input_channels=1,
        base_num_features=base_num_features,
        num_classes=num_classes,
        num_pool=num_pool,
        num_conv_per_stage=2,
        feat_map_mul_on_downscale=2,
        conv_op=nn.Conv3d,
        norm_op=nn.BatchNorm3d,
        norm_op_kwargs={'eps': 1e-5, 'affine': True},
        dropout_op=nn.Dropout3d,
        dropout_op_kwargs={'p': 0.1, 'inplace': True},
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={'negative_slope': 1e-2, 'inplace': True},
        deep_supervision=True,
        net_num_pool_op_kernel_sizes=net_num_pool_op_kernel_sizes,
        net_conv_kernel_sizes=net_conv_kernel_sizes,
        upscale_logits=False,
        convolutional_pool=True,
        convolutional_upsample=True
    )

    nnunet_model = DeepSupervisionWrapper(nnunet_model)

    # Initialize FixMatch trainer
    fixmatch_trainer = FixMatchTrainer(
        model=nnunet_model,
        num_classes=num_classes,
        device=device,
        threshold=0.95,
        lambda_unsup=1.0
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        nnunet_model.parameters(),
        lr=1e-4,
        weight_decay=1e-5
    )

    # Dummy data
    labeled_batch = {
        'data': np.random.randn(2, 1, 128, 128, 64).astype(np.float32),
        'seg': np.random.randint(0, num_classes, (2, 1, 128, 128, 64)).astype(np.int32)
    }

    unlabeled_batch = {
        'data': np.random.randn(4, 1, 128, 128, 64).astype(np.float32)
    }

    # Training loop
    for epoch in range(10):
        metrics = fixmatch_trainer.train_step(labeled_batch, unlabeled_batch, optimizer)

        if (epoch + 1) % 5 == 0:  # Print every 5 epochs
            print(
                f"Epoch {epoch + 1} | Total Loss: {metrics['total_loss']:.4f} "
                f"| Sup Loss: {metrics['supervised_loss']:.4f} "
                f"| Unsup Loss: {metrics['unsupervised_loss']:.4f}")