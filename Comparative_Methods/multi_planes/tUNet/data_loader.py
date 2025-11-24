import os
import yaml
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import zoom
from sklearn.preprocessing import LabelEncoder
import gc
import re


class MultiViewNiftiDataset(Dataset):

    def __init__(self, data_dir, config, transform=None, is_training=True):
        self.data_dir = data_dir
        self.label_dir = config['data']['label_dir']
        self.config = config
        self.transform = transform
        self.is_training = is_training
        self.target_size = tuple(config['data']['target_size'])
        self.n_classes = config['model']['n_classes']
        self.use_pseudo_labels = config['data'].get('use_pseudo_labels', True)
        self.directions = ['sag', 'cor', 'tra']
        self.direction_map = {direction: idx for idx, direction in enumerate(self.directions)}
        self.patient_data = self._organize_patient_data()
        self._analyze_labels()

    def _organize_patient_data(self):
        patient_data = {}
        all_files = os.listdir(self.data_dir)
        for file in all_files:
            if file.endswith(self.config['data']['image_suffix']):
                # UMD_xxxxxx_sag_0000.nii.gz
                match = re.match(r'UMD_(\d+)_(\w+)_0000\.nii\.gz', file)
                if match:
                    patient_id = match.group(1)
                    direction = match.group(2)

                    if direction in self.directions:
                        if patient_id not in patient_data:
                            patient_data[patient_id] = {}

                        image_path = os.path.join(self.data_dir, file)
                        patient_data[patient_id][direction] = {
                            'image': image_path,
                            'direction': direction
                        }

        for patient_id, directions in patient_data.items():
            if 'sag' in directions:
                label_name = f"UMD_{patient_id}_sag{self.config['data']['label_suffix']}"
                label_path = os.path.join(self.label_dir, label_name)

                if os.path.exists(label_path):
                    directions['sag']['label'] = label_path
                else:
                    if self.use_pseudo_labels:
                        print(f"Warning {patient_id} not exist: {label_path}")
                    else:
                        print(f"Patient {patient_id} label not exist")
                        del patient_data[patient_id]
                        continue

        complete_patients = {}
        for patient_id, directions in patient_data.items():
            if len(directions) == 3:
                if not self.use_pseudo_labels or ('sag' in directions and 'label' in directions['sag']):
                    complete_patients[patient_id] = directions

        return complete_patients

    def _analyze_labels(self):
        all_labels = set()

        for patient_id, directions in self.patient_data.items():
            if 'sag' in directions and 'label' in directions['sag']:
                try:
                    label_nib = nib.load(directions['sag']['label'])
                    label_data = label_nib.get_fdata()
                    unique_labels = np.unique(label_data)
                    all_labels.update(unique_labels)

                    del label_nib, label_data
                    gc.collect()

                except Exception as e:
                    print(f"{directions['sag']['label']} error: {e}")

        self.label_classes = sorted(list(all_labels))

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.label_classes)

    def __len__(self):
        return len(self.patient_data)

    def __getitem__(self, idx):
        try:
            patient_ids = list(self.patient_data.keys())
            patient_id = patient_ids[idx]
            directions = self.patient_data[patient_id]
            sag_data = self._load_direction_data(directions['sag'], 'sag')
            cor_data = self._load_direction_data(directions['cor'], 'cor')
            tra_data = self._load_direction_data(directions['tra'], 'tra')

            if 'sag' in directions and 'label' in directions['sag']:
                label_nib = nib.load(directions['sag']['label'])
                label_data = label_nib.get_fdata().astype(np.uint8)
                label_data = self.preprocess_label(label_data)
                label_tensor = torch.from_numpy(label_data).long()
            else:
                label_tensor = torch.zeros(self.target_size, dtype=torch.long)

            sag_tensor = torch.from_numpy(sag_data).unsqueeze(0)
            cor_tensor = torch.from_numpy(cor_data).unsqueeze(0)
            tra_tensor = torch.from_numpy(tra_data).unsqueeze(0)

            if self.transform and self.is_training:
                sag_tensor, label_tensor = self.transform(sag_tensor, label_tensor)
                cor_tensor, _ = self.transform(cor_tensor, label_tensor)
                tra_tensor, _ = self.transform(tra_tensor, label_tensor)

            return {
                'sag': sag_tensor,
                'cor': cor_tensor,
                'tra': tra_tensor,
                'label': label_tensor,
                'patient_id': patient_id,
                'sag_path': directions['sag']['image'],
                'cor_path': directions['cor']['image'],
                'tra_path': directions['tra']['image']
            }

        except Exception as e:
            empty_image = torch.zeros((1,) + self.target_size, dtype=torch.float32)
            empty_label = torch.zeros(self.target_size, dtype=torch.long)
            return {
                'sag': empty_image,
                'cor': empty_image,
                'tra': empty_image,
                'label': empty_label,
                'patient_id': f'error_{idx}',
                'sag_path': '',
                'cor_path': '',
                'tra_path': ''
            }

    def _load_direction_data(self, direction_info, direction_name):
        try:
            image_nib = nib.load(direction_info['image'])
            image_data = image_nib.get_fdata().astype(np.float32)
            image_data = self.preprocess_image(image_data)
            del image_nib
            return image_data

        except Exception as e:
            print(f"{direction_name}error: {e}")
            return np.zeros(self.target_size, dtype=np.float32)

    def preprocess_image(self, image_data):
        # [0,1]
        if image_data.max() > 0:
            image_data = (image_data - image_data.min()) / (image_data.max() - image_data.min())

        if image_data.shape != self.target_size:
            zoom_factors = [self.target_size[i] / image_data.shape[i] for i in range(3)]
            image_data = zoom(image_data, zoom_factors, order=1)

        return image_data

    def preprocess_label(self, label_data):
        if label_data.shape != self.target_size:
            zoom_factors = [self.target_size[i] / label_data.shape[i] for i in range(3)]
            label_data = zoom(label_data, zoom_factors, order=0)

        label_data = np.clip(label_data, 0, self.n_classes - 1)

        return label_data.astype(np.uint8)


class ContrastiveDataLoader:

    def __init__(self, dataset, batch_size, shuffle=True, num_workers=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            np.random.shuffle(indices)

        # batch_size
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch_data = []

            for idx in batch_indices:
                data = self.dataset[idx]
                batch_data.append(data)

            batch = self._collate_fn(batch_data)
            yield batch

    def _collate_fn(self, batch_data):
        sag_images = torch.stack([data['sag'] for data in batch_data])
        cor_images = torch.stack([data['cor'] for data in batch_data])
        tra_images = torch.stack([data['tra'] for data in batch_data])
        labels = torch.stack([data['label'] for data in batch_data])
        patient_ids = [data['patient_id'] for data in batch_data]

        return {
            'sag': sag_images,
            'cor': cor_images,
            'tra': tra_images,
            'label': labels,
            'patient_id': patient_ids
        }


def get_multi_view_data_loaders(config, transform=None):
    data_dir = config['data']['data_dir']
    batch_size = config['training']['batch_size']
    train_split = config['data']['train_split']
    full_dataset = MultiViewNiftiDataset(data_dir, config, transform, is_training=True)
    label_classes = full_dataset.label_classes
    dataset_size = len(full_dataset)
    train_size = int(train_split * dataset_size)
    val_size = dataset_size - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    train_loader = ContrastiveDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config['misc']['num_workers']
    )

    val_loader = ContrastiveDataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config['misc']['num_workers']
    )

    return train_loader, val_loader, label_classes

class NiftiDataset(Dataset):
    def __init__(self, data_dir, config, transform=None, is_training=True):
        self.data_dir = data_dir
        self.config = config
        self.transform = transform
        self.is_training = is_training
        self.target_size = tuple(config['data']['target_size'])
        self.n_classes = config['model']['n_classes']
        self.image_files = []
        self.label_files = []

        for file in os.listdir(data_dir):
            if file.endswith(config['data']['image_suffix']):
                # name_0000.nii.gz
                image_path = os.path.join(data_dir, file)
                # name.nii.gz
                label_name = file.replace(config['data']['image_suffix'], config['data']['label_suffix'])
                label_path = os.path.join(data_dir, label_name)

                if os.path.exists(label_path):
                    self.image_files.append(image_path)
                    self.label_files.append(label_path)
        self._analyze_labels()

    def _analyze_labels(self):
        all_labels = set()

        for i, label_file in enumerate(self.label_files):
            if i % 5 == 0:
                print(f"Analyze {i}/{len(self.label_files)}")

            try:
                label_nib = nib.load(label_file)
                label_data = label_nib.get_fdata()
                unique_labels = np.unique(label_data)
                all_labels.update(unique_labels)

                del label_nib, label_data
                gc.collect()

            except Exception as e:
                print(f" {label_file} error: {e}")

        self.label_classes = sorted(list(all_labels))
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.label_classes)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        try:
            image_nib = nib.load(self.image_files[idx])
            image_data = image_nib.get_fdata().astype(np.float32)
            label_nib = nib.load(self.label_files[idx])
            label_data = label_nib.get_fdata().astype(np.uint8)
            image_data = self.preprocess_image(image_data)
            label_data = self.preprocess_label(label_data)
            image_tensor = torch.from_numpy(image_data).unsqueeze(0)
            label_tensor = torch.from_numpy(label_data).long()

            if self.transform and self.is_training:
                image_tensor, label_tensor = self.transform(image_tensor, label_tensor)
            del image_nib, label_nib, image_data, label_data
            return image_tensor, label_tensor

        except Exception as e:
            print(f"Error  {idx}): {e}")
            empty_image = torch.zeros((1,) + self.target_size, dtype=torch.float32)
            empty_label = torch.zeros(self.target_size, dtype=torch.long)
            return empty_image, empty_label

    def preprocess_image(self, image_data):
        if image_data.max() > 0:
            image_data = (image_data - image_data.min()) / (image_data.max() - image_data.min())

        if image_data.shape != self.target_size:
            zoom_factors = [self.target_size[i] / image_data.shape[i] for i in range(3)]
            image_data = zoom(image_data, zoom_factors, order=1)

        return image_data

    def preprocess_label(self, label_data):
        if label_data.shape != self.target_size:
            zoom_factors = [self.target_size[i] / label_data.shape[i] for i in range(3)]
            label_data = zoom(label_data, zoom_factors, order=0)
        label_data = np.clip(label_data, 0, self.n_classes - 1)
        return label_data.astype(np.uint8)


def get_data_loaders(config, transform=None):
    data_dir = config['data']['data_dir']
    batch_size = config['training']['batch_size']
    train_split = config['data']['train_split']
    full_dataset = NiftiDataset(data_dir, config, transform, is_training=True)
    label_classes = full_dataset.label_classes
    dataset_size = len(full_dataset)
    train_size = int(train_split * dataset_size)
    val_size = dataset_size - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config['misc']['num_workers'],
        pin_memory=config['misc']['pin_memory'],
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config['misc']['num_workers'],
        pin_memory=config['misc']['pin_memory'],
        drop_last=True
    )

    return train_loader, val_loader, label_classes


def load_config(config_path="config.yaml"):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config