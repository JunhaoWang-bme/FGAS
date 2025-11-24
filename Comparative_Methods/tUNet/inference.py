#!/usr/bin/env python3

import os
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import nibabel as nib
from tqdm import tqdm
import argparse
from model import get_model
from data_loader import MultiViewNiftiDataset


class MultiViewInference:
    def __init__(self, config, model_path, device='auto'):
        self.config = config
        self.device = self._setup_device(device)
        self.model = self._load_model(model_path)
        self.model.eval()
        self.target_size = tuple(config['data']['target_size'])
        self.n_classes = config['model']['n_classes']

    def _setup_device(self, device):
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return torch.device(device)

    def _load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        model = get_model(self.config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)

        return model

    def preprocess_image(self, image_data):
        # [0,1]
        if image_data.max() > 0:
            image_data = (image_data - image_data.min()) / (image_data.max() - image_data.min())

        if image_data.shape != self.target_size:
            from scipy.ndimage import zoom
            zoom_factors = [self.target_size[i] / image_data.shape[i] for i in range(3)]
            image_data = zoom(image_data, zoom_factors, order=1)

        return image_data

    def predict_single_direction(self, image_path, direction_idx):
        try:
            image_nib = nib.load(image_path)
            image_data = image_nib.get_fdata().astype(np.float32)
            original_shape = image_data.shape
            processed_data = self.preprocess_image(image_data)
            image_tensor = torch.from_numpy(processed_data).unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
            image_tensor = image_tensor.to(self.device)

            with torch.no_grad():
                if hasattr(self.model, 'forward') and 'direction_idx' in self.model.forward.__code__.co_varnames:
                    logits = self.model(image_tensor, direction_idx=direction_idx)
                else:
                    logits = self.model(image_tensor)

                if logits.dim() == 5:  # [B, C, D, H, W]
                    probs = F.softmax(logits, dim=1)
                    predictions = torch.argmax(probs, dim=1)
                else:
                    return None, None, None

            predictions = predictions.cpu().numpy().squeeze()
            probs = probs.cpu().numpy().squeeze()

            if predictions.shape != original_shape:
                from scipy.ndimage import zoom
                zoom_factors = [original_shape[i] / predictions.shape[i] for i in range(3)]
                print(f"  Zoom: {zoom_factors}")

                if len(zoom_factors) != len(predictions.shape):
                    print(f" Warning: zoom_factors ({len(zoom_factors)}) ({len(predictions.shape)})")
                    if len(predictions.shape) == 3:
                        zoom_factors = zoom_factors[:3]
                    else:
                        return None, None, None

                predictions = zoom(predictions, zoom_factors, order=0)

                if probs.ndim == 4:  # [C, D, H, W]
                    zoomed_probs = np.zeros((probs.shape[0],) + original_shape, dtype=probs.dtype)
                    for c in range(probs.shape[0]):
                        zoomed_probs[c] = zoom(probs[c], zoom_factors, order=1)
                    probs = zoomed_probs
                elif probs.ndim == 3:  # [D, H, W]
                    probs = zoom(probs, zoom_factors, order=1)
                else:
                    probs = np.zeros((self.n_classes,) + original_shape, dtype=np.float32)
                    probs[0] = 1.0

            return predictions.astype(np.uint8), probs, original_shape

        except Exception as e:
            print(f"No {image_path}: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None

    def predict_multi_view(self, sag_path, cor_path, tra_path):

        sag_pred, sag_probs, sag_shape = self.predict_single_direction(sag_path, direction_idx=0)
        cor_pred, cor_probs, cor_shape = self.predict_single_direction(cor_path, direction_idx=1)
        tra_pred, tra_probs, tra_shape = self.predict_single_direction(tra_path, direction_idx=2)

        if sag_pred is None or cor_pred is None or tra_pred is None:
            return None

        return {
            'sag': {'pred': sag_pred, 'probs': sag_probs, 'shape': sag_shape},
            'cor': {'pred': cor_pred, 'probs': cor_probs, 'shape': cor_shape},
            'tra': {'pred': tra_pred, 'probs': tra_probs, 'shape': tra_shape}
        }

    def save_predictions(self, predictions, output_dir, patient_id):
        os.makedirs(output_dir, exist_ok=True)

        for direction, result in predictions.items():
            pred_path = os.path.join(output_dir, f"{patient_id}_{direction}_pred.nii.gz")
            pred_nii = nib.Nifti1Image(result['pred'], np.eye(4))
            nib.save(pred_nii, pred_path)
            print(f"Save: {pred_path}")

            probs = result['probs']
            if probs.ndim == 4:  # [C, D, H, W]
                for class_idx in range(min(probs.shape[0], self.n_classes)):
                    prob_path = os.path.join(output_dir, f"{patient_id}_{direction}_prob_class{class_idx}.nii.gz")
                    prob_nii = nib.Nifti1Image(probs[class_idx], np.eye(4))
                    nib.save(prob_nii, prob_path)
            elif probs.ndim == 3:  # [D, H, W]
                prob_path = os.path.join(output_dir, f"{patient_id}_{direction}_prob.nii.gz")
                prob_nii = nib.Nifti1Image(probs, np.eye(4))
                nib.save(prob_nii, prob_path)
            else:
                print(f"Warning: {probs.shape}")

            if probs.ndim == 4:  # [C, D, H, W]
                max_prob = np.max(probs, axis=0)
            elif probs.ndim == 3:  # [D, H, W]
                max_prob = probs
            else:
                max_prob = np.zeros(result['pred'].shape, dtype=np.float32)
                max_prob[result['pred'] > 0] = 1.0

            max_prob_path = os.path.join(output_dir, f"{patient_id}_{direction}_max_prob.nii.gz")
            max_prob_nii = nib.Nifti1Image(max_prob, np.eye(4))
            nib.save(max_prob_nii, max_prob_path)

    def batch_inference(self, data_dir, output_dir):

        dataset = MultiViewNiftiDataset(data_dir, self.config, transform=None, is_training=False)
        successful_predictions = 0
        failed_predictions = 0

        for i in tqdm(range(len(dataset)), desc="continuing"):
            try:
                sample = dataset[i]
                patient_id = sample['patient_id']
                sag_path = sample['sag_path'] if 'sag_path' in sample else None
                cor_path = sample['cor_path'] if 'cor_path' in sample else None
                tra_path = sample['tra_path'] if 'tra_path' in sample else None

                if sag_path and cor_path and tra_path:
                    predictions = self.predict_multi_view(sag_path, cor_path, tra_path)

                    if predictions:
                        patient_output_dir = os.path.join(output_dir, patient_id)
                        self.save_predictions(predictions, patient_output_dir, patient_id)
                        self._print_statistics(predictions, patient_id)

                        successful_predictions += 1
                    else:
                        failed_predictions += 1
                else:
                    failed_predictions += 1

            except Exception as e:
                print(f" {patient_id} error: {e}")
                failed_predictions += 1
                continue

    def _print_statistics(self, predictions, patient_id):

        for direction, result in predictions.items():
            pred = result['pred']
            unique_labels, counts = np.unique(pred, return_counts=True)

            print(f"  {direction.upper()}:")
            for label, count in zip(unique_labels, counts):
                percentage = count / pred.size * 100
                print(f"  {int(label)}: {count} ({percentage:.2f}%)")


def main():
    """主函数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--output_dir', type=str, default='predictions')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--single_case', action='store_true')
    parser.add_argument('--sag_path', type=str)
    parser.add_argument('--cor_path', type=str)
    parser.add_argument('--tra_path', type=str)

    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    inferencer = MultiViewInference(config, args.model, args.device)

    if args.single_case:
        if not all([args.sag_path, args.cor_path, args.tra_path]):
            return

        predictions = inferencer.predict_multi_view(args.sag_path, args.cor_path, args.tra_path)
        if predictions:
            inferencer.save_predictions(predictions, args.output_dir, "single_case")
            inferencer._print_statistics(predictions, "single_case")
    else:
        if not args.data_dir:
            return

        inferencer.batch_inference(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()

