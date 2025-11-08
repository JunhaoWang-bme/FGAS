import sys
import os
sys.path.append("/home/login/Documents/wjh_proj_myoma/Code/nnunet_with_nninteractive")
import os
import torch
import numpy as np
from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
from skimage.measure import label, regionprops
import time
from typing import List, Tuple
from scipy.ndimage import label as nd_label


# ---------------------- Refiner ----------------------
class Refiner:
    def __init__(self,
                 model_folder="/home/login/Documents/nnInteractive/nnInteractive_v1.0",
                 device="cuda:3"):
        self.bbox_shrink = 5
        self.border_threshold = 3
        self.session = nnInteractiveInferenceSession(
            device=torch.device(device),
            use_torch_compile=False,
            verbose=False,
            torch_n_threads=os.cpu_count(),
            do_autozoom=True,
            use_pinned_memory=True,
        )
        self.session.initialize_from_trained_model_folder(model_folder)
        self.device = device
        self.label_color_map = {1: [0, 0, 255], 2: [0, 255, 0], 3: [255, 0, 0], 4: [0, 255, 255]}

    def generate_unique_case_id(self):
        timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
        ms = int(time.time() * 1000) % 1000
        rand = np.random.randint(0, 100)
        return f"case_{timestamp}_{ms:03d}_{rand:02d}"

    def extract_points_from_mask(self, mask, num_points=5):
        coords = np.argwhere(mask == 1)
        if len(coords) == 0:
            return []
        num_points = min(num_points, len(coords))
        center = coords.mean(axis=0).astype(int)
        points = [tuple(center)]
        if num_points > 1:
            coords = coords[~np.all(coords == center, axis=1)]
            if len(coords) > 0:
                indices = np.random.choice(len(coords), size=num_points - 1, replace=False)
                sampled = coords[indices]
                points.extend([tuple(p) for p in sampled])
        return points


    def refine(self, output: torch.Tensor, image_tensor: torch.Tensor, case_id: str = None) -> torch.Tensor:
        self._check_input_validity(output, image_tensor)
        B, C, D, H, W = output.shape
        case_id = case_id or self._generate_unique_case_id()

        pred_mask = self._logits_to_mask(output)

        img_np = image_tensor[0].cpu().numpy()
        mask_initial_np = pred_mask[0].cpu().numpy()

        if np.count_nonzero(img_np) == 0 or np.count_nonzero(mask_initial_np) == 0:
            return pred_mask.unsqueeze(1)

        mask_refined_np = np.zeros_like(mask_initial_np)
        img_xyz = img_np.transpose(2, 1, 0)
        mask_initial_xyz = mask_initial_np.transpose(2, 1, 0)

        for label_id in [1, 4, 2, 3]:
            label_mask_xyz = (mask_initial_xyz == label_id).astype(np.uint8)
            label_total_pixels = np.count_nonzero(label_mask_xyz)
            if label_total_pixels == 0:
                mask_refined_np[mask_initial_np == label_id] = label_id
                continue

            bbox_2d = []
            max_points = 5 if label_id != 4 else 2
            key_z = self._get_label_key_slice(label_mask_xyz) if label_id != 4 else None
            if label_id != 4 and key_z is None:
                mask_refined_np[mask_initial_np == label_id] = label_id
                continue

            if label_id != 4:
                key_slice_mask = label_mask_xyz[:, :, key_z]
                img_width, img_height = key_slice_mask.shape[0], key_slice_mask.shape[1]
                bbox_2d = self._get_single_slice_bbox(key_slice_mask, key_z, img_width, img_height)

            all_points_3d = self._get_points_from_all_regions(label_mask_xyz, max_points=max_points)
            if not all_points_3d:
                mask_refined_np[mask_initial_np == label_id] = label_id
                continue

            label_refined_xyz = self._optimize_with_nninteractive(img_xyz, bbox_2d, all_points_3d, label_id)
            mask_refined_np[label_refined_xyz.transpose(2, 1, 0) > 0] = label_id

        refined_target = torch.from_numpy(mask_refined_np).unsqueeze(0).unsqueeze(1).long().to(self.device)
        return refined_target

    def _check_input_validity(self, output: torch.Tensor, image_tensor: torch.Tensor):
        assert output.ndim == 5, f"output error, but: {output.ndim}"
        assert image_tensor.ndim == 4, f"image_tensor error, but: {image_tensor.ndim}"
        assert output.shape[0] == 1, "B=1"
        assert output.shape[1] >= 5, f">=5，but: {output.shape[1]}"

    def _logits_to_mask(self, output: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(output, dim=1)
        return torch.argmax(probs, dim=1)

    def _generate_unique_case_id(self) -> str:
        timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
        ms = int(time.time() * 1000) % 1000
        rand = np.random.randint(0, 100)
        return f"case_{timestamp}_{ms:03d}_{rand:02d}"

    def _get_label_key_slice(self, label_mask_xyz: np.ndarray) -> int:
        total_z = label_mask_xyz.shape[2]
        z_pixel_count = [(z, np.count_nonzero(label_mask_xyz[:, :, z])) for z in range(total_z)]
        z_pixel_count.sort(key=lambda x: (-x[1], x[0]))
        valid_slices = [z for z, cnt in z_pixel_count if cnt > 0]
        return valid_slices[0] if valid_slices else None

    def _get_single_slice_bbox(self, slice_mask_2d: np.ndarray, key_z: int, img_width: int, img_height: int) -> List[List[int]]:
        coords = np.argwhere(slice_mask_2d == 1)
        if len(coords) == 0:
            return []

        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)
        orig_x_min, orig_x_max = x_min, x_max
        orig_y_min, orig_y_max = y_min, y_max

        if x_min <= self.border_threshold:
            x_min = min(x_max - 1, x_min + self.bbox_shrink)
        if x_max >= img_width - self.border_threshold:
            x_max = max(x_min + 1, x_max - self.bbox_shrink)
        if y_min <= self.border_threshold:
            y_min = min(y_max - 1, y_min + self.bbox_shrink)
        if y_max >= img_height - self.border_threshold:
            y_max = max(y_min + 1, y_max - self.bbox_shrink)

        if x_min >= x_max:
            x_min, x_max = orig_x_min, orig_x_max
        if y_min >= y_max:
            y_min, y_max = orig_y_min, orig_y_max

        return [[x_min, x_max + 1], [y_min, y_max + 1], [key_z, key_z + 1]]

    def _get_points_from_all_regions(self, label_mask_xyz: np.ndarray, max_points: int = 5) -> List[Tuple[int, int, int]]:
        labeled_regions, num_regions = nd_label(label_mask_xyz)
        if num_regions == 0:
            return []
        region_centers = []
        for rid in range(1, num_regions + 1):
            coords = np.argwhere(labeled_regions == rid)
            if len(coords) == 0:
                continue
            center = tuple(coords.mean(axis=0).astype(int))
            region_centers.append(center)
        return list(dict.fromkeys(region_centers))[:max_points]

    def _optimize_with_nninteractive(self, img_xyz: np.ndarray, bbox_2d: List[List[int]],
                                     points_3d: List[Tuple[int, int, int]], label_id: int) -> np.ndarray:
        try:
            self.session.reset_interactions()
            self.session.set_image(img_xyz[None])
            target_buf = torch.zeros_like(torch.from_numpy(img_xyz)).to(self.device)
            self.session.set_target_buffer(target_buf)
            if bbox_2d:
                self.session.add_bbox_interaction(bbox_2d, include_interaction=True)
            for point in points_3d:
                self.session.add_point_interaction(point, include_interaction=True)
            return self.session.target_buffer.clone().cpu().numpy()
        except Exception:
            return (img_xyz == label_id).astype(np.uint8)

    def _preprocess_image(self, img_np: np.ndarray) -> np.ndarray:
        D, H, W = img_np.shape
        img_min, img_max = img_np.min(), img_np.max()
        if img_max - img_min < 1e-8:
            img_norm = np.zeros_like(img_np, dtype=np.uint8)
        else:
            img_norm = ((img_np - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        return np.stack([img_norm, img_norm, img_norm], axis=-1)



def find_max_info_slice(mask_region):
    z_slices_nonzero = [np.sum(mask_region[:, :, z]) for z in range(mask_region.shape[2])]
    max_z_index = np.argmax(z_slices_nonzero)
    max_nonzero_count = z_slices_nonzero[max_z_index]
    return (mask_region.shape[2] // 2, 0) if max_nonzero_count == 0 else (max_z_index, max_nonzero_count)


def extract_largest_connected_component(mask):
    if np.sum(mask) == 0:
        return mask
    labeled_mask = label(mask)
    region_props = regionprops(labeled_mask)
    if not region_props:
        return np.zeros_like(mask)
    largest_region = max(region_props, key=lambda r: r.area)
    return (labeled_mask == largest_region.label).astype(np.uint8)


def extract_plane_bboxes_by_label(mask_np):
    label_bboxes = {}
    expand_params = {1: 0, 3: 0, 2: 0, 4: 0}
    for label_id in [1, 2, 3, 4]:
        binary_mask = (mask_np == label_id).astype(np.uint8)
        if np.sum(binary_mask) == 0:
            continue
        labeled = label(binary_mask)
        props = regionprops(labeled)
        bbox_z_list = []
        for region in props:
            region_mask = np.zeros_like(binary_mask)
            region_mask[labeled == region.label] = 1
            z_max_info, _ = find_max_info_slice(region_mask)
            slice_mask = region_mask[:, :, z_max_info]
            if np.sum(slice_mask) == 0:
                zmin, ymin, xmin, zmax, ymax, xmax = region.bbox
                z_max_info = (zmin + zmax) // 2
                slice_mask = region_mask[:, :, z_max_info]
            non_zero_coords = np.argwhere(slice_mask > 0)
            if len(non_zero_coords) == 0:
                continue
            x_coords, y_coords = non_zero_coords[:, 0], non_zero_coords[:, 1]
            xmin, xmax = np.min(x_coords), np.max(x_coords) + 1
            ymin, ymax = np.min(y_coords), np.max(y_coords) + 1
            expand_pixels = expand_params.get(label_id, 0)
            if expand_pixels > 0:
                max_x, max_y = slice_mask.shape[0], slice_mask.shape[1]
                xmin = max(0, xmin - expand_pixels)
                xmax = min(max_x, xmax + expand_pixels)
                ymin = max(0, ymin - expand_pixels)
                ymax = min(max_y, ymax + expand_pixels)
            bbox = [[xmin, xmax], [ymin, ymax], [z_max_info, z_max_info + 1]]
            bbox_z_list.append((bbox, z_max_info))
        if bbox_z_list:
            label_bboxes[label_id] = bbox_z_list
    return label_bboxes
