from collections import defaultdict
from torch.cuda.amp import autocast
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.training.dataloading.umd_dataloader import UMDDataLoader3D, UMDDataLoader2D, UMDSequentialDataLoader
from nnunet.training.loss_functions.multiview_consistency_loss import create_multiview_consistency_loss
import os
import torch
import numpy as np
from .refiner import Refiner


class UMDTrainer(nnUNetTrainer):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, view_aware_training=False):
        super(UMDTrainer, self).__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage,
                                         unpack_data, deterministic, fp16)

        self.view_aware_training = view_aware_training
        self.view_statistics = defaultdict(int)
        self.patient_statistics = defaultdict(int)

    def setup_DA_params(self):
        super(UMDTrainer, self).setup_DA_params()
        self.data_aug_params['num_threads'] = 0
        self.data_aug_params['num_cached_per_thread'] = 1

        if self.view_aware_training:
            self.view_specific_aug_params = {
                'sag': {'rotation_x': (-10, 10), 'rotation_y': (-5, 5), 'rotation_z': (-15, 15)},
                'cor': {'rotation_x': (-10, 10), 'rotation_y': (-5, 5), 'rotation_z': (-15, 15)},
                'tra': {'rotation_x': (-10, 10), 'rotation_y': (-5, 5), 'rotation_z': (-15, 15)}
            }

    def get_basic_generators(self):
        self.load_dataset()
        self.do_split()

        if self.threeD:
            dl_tr = UMDDataLoader3D(self.dataset_tr, self.basic_generator_patch_size, self.patch_size, self.batch_size,
                                    self.oversample_foreground_percent, "r", 1, "constant", None, self.pad_all_sides)
            dl_val = UMDDataLoader3D(self.dataset_val, self.patch_size, self.patch_size, self.batch_size,
                                     self.oversample_foreground_percent, "r", 1, "constant", None, self.pad_all_sides)
        else:
            dl_tr = UMDDataLoader2D(self.dataset_tr, self.basic_generator_patch_size, self.patch_size, self.batch_size,
                                    self.oversample_foreground_percent, "r", 1, "constant", None, self.pad_all_sides)
            dl_val = UMDDataLoader2D(self.dataset_val, self.patch_size, self.patch_size, self.batch_size,
                                     self.oversample_foreground_percent, "r", 1, "constant", None, self.pad_all_sides)

        return dl_tr, dl_val


class UMDViewAwareTrainer(UMDTrainer):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super(UMDViewAwareTrainer, self).__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage,
                                                  unpack_data, deterministic, fp16, view_aware_training=True)

        self.view_learning_rates = {'sag': 1e-4, 'cor': 1e-4, 'tra': 1e-4}
        self.view_loss_weights = {'sag': 2.0, 'cor': 1.0, 'tra': 1.0}

    def generate_train_batch(self):
        data_dict = super(UMDSequentialDataLoader, self).generate_train_batch()
        case_info_list = []

        for i in data_dict['idx']:
            props = self._data[i]['properties']
            case_id = props.get('case_identifier', f'case_{i}')
            image_path = props['image_filenames'][0] if (
                        'image_filenames' in props and props['image_filenames']) else None

            case_info_list.append({'case_id': case_id, 'image_path': image_path})

        data_dict['case_info'] = case_info_list
        return data_dict

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']

        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target).float()

        device = next(self.network.parameters()).device
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        case_info = data_dict.get('case_info', [])
        for info in case_info:
            if isinstance(info, dict) and 'view' in info:
                self.view_statistics[info['view']] += 1

        self.optimizer.zero_grad()

        if self.fp16:
            with autocast():
                output = self.network(data)
                del data
                l = self.loss(output, target)
        else:
            output = self.network(data)
            del data
            l = self.loss(output, target)

        if do_backprop:
            if self.fp16:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
            else:
                l.backward()
                self.optimizer.step()

            self.optimizer.zero_grad()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target
        return l.detach().cpu().numpy()


class UMDConsistencyTrainer(UMDTrainer):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, consistency_weight=1.0, consistency_type='dice'):
        super(UMDConsistencyTrainer, self).__init__(plans_file, fold, output_folder, dataset_directory, batch_dice,
                                                    stage, unpack_data, deterministic, fp16)

        # Replace it with its own weight path
        model_folder = "/home/login/Documents/nnInteractive/nnInteractive_v1.0"
        self.refiner = Refiner(model_folder, device="cuda:0")
        self.consistency_weight = consistency_weight
        self.consistency_type = consistency_type
        self.consistency_loss_fn = create_multiview_consistency_loss(
            base_loss_fn=None,
            consistency_weight=consistency_weight,
            consistency_type=consistency_type
        )

    def generate_train_batch(self):
        data_dict = super(UMDSequentialDataLoader, self).generate_train_batch()
        case_info_list = []

        for i in data_dict['idx']:
            props = self._data[i]['properties']
            case_id = props.get('case_identifier')
            if case_id is None and 'image_filenames' in props and props['image_filenames']:
                fname = os.path.basename(props['image_filenames'][0])
                case_id = os.path.splitext(fname)[0]
            case_id = case_id or f"case_{i}"

            image_path = props['image_filenames'][0] if (
                        'image_filenames' in props and props['image_filenames']) else None
            case_info_list.append({'case_id': case_id, 'image_path': image_path})

        data_dict['case_info'] = case_info_list
        return data_dict

    def get_basic_generators(self):
        self.load_dataset()
        self.do_split()

        if self.threeD:
            dl_tr = UMDSequentialDataLoader(
                self.dataset_tr, self.basic_generator_patch_size, self.patch_size, self.batch_size,
                False, self.oversample_foreground_percent, "r", "constant", None, self.pad_all_sides
            )
            dl_val = UMDDataLoader3D(
                self.dataset_val, self.patch_size, self.patch_size, self.batch_size,
                self.oversample_foreground_percent, "r", 1, "constant", None, self.pad_all_sides
            )
        else:
            dl_tr = UMDSequentialDataLoader(
                self.dataset_tr, self.basic_generator_patch_size, self.patch_size, self.batch_size,
                False, self.oversample_foreground_percent, "r", "constant", None, self.pad_all_sides
            )
            dl_val = UMDDataLoader2D(
                self.dataset_val, self.patch_size, self.patch_size, self.batch_size,
                self.oversample_foreground_percent, "r", 1, "constant", None, self.pad_all_sides
            )

        return dl_tr, dl_val

    def setup_DA_params(self):
        super(UMDConsistencyTrainer, self).setup_DA_params()
        self.data_aug_params['num_threads'] = 2
        self.data_aug_params['num_threads'] = 2
        self.data_aug_params['num_cached_per_thread'] = 1

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']

        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target).float()

        device = next(self.network.parameters()).device
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        case_info = data_dict.get('case_info', [])
        consistency_pairs = data_dict.get('consistency_pairs', [])

        self.optimizer.zero_grad()
        B = data.shape[0]
        refined_targets = []

        if self.fp16:
            with autocast():
                output = self.network(data)
                for b in range(B):
                    if b >= len(case_info):
                        refined_targets.append(torch.argmax(output[b:b + 1], dim=1).long())
                        continue
                    case_id = case_info[b].get('case_id', f'case_{b}')
                    refined_mask_b = self.refiner.refine(output[b:b + 1], data[b:b + 1].squeeze(1), case_id=None)

                    dice = compute_meandice(
                        refined_mask_b.cpu().numpy(),
                        output[b:b + 1].detach().cpu().numpy()
                    )[0]
                    base_dice = compute_meandice(
                        output[b:b + 1].detach().argmax(1).cpu().numpy(),
                        target[b:b + 1].cpu().numpy()
                    )[0]

                    if dice >= max(base_dice - 0.1, 0.5):
                        refined_targets.append(refined_mask_b)
                    else:
                        refined_targets.append(target[b:b + 1].long())

                refined_target = torch.cat(refined_targets, dim=0)
                del data
                l, loss_dict = self.consistency_loss_fn(output, refined_target, case_info, consistency_pairs)
        else:
            output = self.network(data)
            for b in range(B):
                if b >= len(case_info):
                    refined_targets.append(torch.argmax(output[b:b + 1], dim=1).long())
                    continue
                case_id = case_info[b].get('case_id', f'case_{b}')
                refined_mask_b = self.refiner.refine(output[b:b + 1], data[b:b + 1].squeeze(1), case_id=None)

                dice = compute_meandice(
                    refined_mask_b.cpu().numpy(),
                    output[b:b + 1].detach().cpu().numpy()
                )[0]
                base_dice = compute_meandice(
                    output[b:b + 1].detach().argmax(1).cpu().numpy(),
                    target[b:b + 1].cpu().numpy()
                )[0]

                if dice >= max(base_dice - 0.1, 0.5):
                    refined_targets.append(refined_mask_b)
                else:
                    refined_targets.append(target[b:b + 1].long())

            refined_target = torch.cat(refined_targets, dim=0)
            del data
            l, loss_dict = self.consistency_loss_fn(output, refined_target, case_info, consistency_pairs)

        if do_backprop:
            if self.fp16:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
            else:
                l.backward()
                self.optimizer.step()

            self.optimizer.zero_grad()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target
        return l.detach().cpu().numpy()


def compute_meandice(prediction: np.ndarray, target: np.ndarray, smooth: float = 1e-5) -> np.ndarray:
    prediction = np.asarray(prediction)
    target = np.asarray(target)

    max_ndim = max(prediction.ndim, target.ndim)
    while prediction.ndim < max_ndim:
        prediction = np.expand_dims(prediction, axis=1)
    while target.ndim < max_ndim:
        target = np.expand_dims(target, axis=1)

    min_shape = [min(p, t) for p, t in zip(prediction.shape[2:], target.shape[2:])]
    pred_slice = (slice(None), slice(None)) + tuple(slice(0, s) for s in min_shape)
    targ_slice = (slice(None), slice(None)) + tuple(slice(0, s) for s in min_shape)

    prediction = prediction[pred_slice]
    target = target[targ_slice]

    batch_size = prediction.shape[0]
    dice_scores = []

    for b in range(batch_size):
        pred = prediction[b].reshape(prediction[b].shape[0], -1)
        targ = target[b].reshape(target[b].shape[0], -1)

        intersection = np.sum(pred * targ, axis=1)
        union = np.sum(pred, axis=1) + np.sum(targ, axis=1)
        per_channel_dice = (2. * intersection + smooth) / (union + smooth)
        dice_scores.append(np.mean(per_channel_dice))

    return np.array(dice_scores)