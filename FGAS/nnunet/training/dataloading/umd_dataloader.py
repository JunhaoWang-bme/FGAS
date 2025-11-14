import re
import numpy as np
from collections import OrderedDict, defaultdict
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.training.dataloading.dataset_loading import DataLoader3D, DataLoader2D


class UMDDataLoader3D(DataLoader3D):
    def __init__(self, data, patch_size, final_patch_size, batch_size, oversample_foreground_percent=0.0,
                 memmap_mode="r", pseudo_3d_slices=1, pad_mode="edge",
                 pad_kwargs_data=None, pad_sides=None):
        super(UMDDataLoader3D, self).__init__(data, patch_size, final_patch_size, batch_size,
                                              False, oversample_foreground_percent, memmap_mode,
                                              pad_mode, pad_kwargs_data, pad_sides)
        self.pseudo_3d_slices = pseudo_3d_slices

        self.case_info = {}
        for key in self._data.keys():
            self.case_info[key] = self._parse_case_info(key)

        self.patient_groups = self._group_by_patient()

    def _parse_case_info(self, case_key):
        match = re.match(r'UMD_(\d+)_(sag|cor|tra)', case_key)
        if match:
            return {
                'case_key': case_key,
                'id': match.group(1),
                'view': match.group(2),
                'view_code': self._get_view_code(match.group(2))
            }
        else:
            return {'case_key': case_key, 'id': 'unknown', 'view': 'unknown', 'view_code': 0}

    def _get_view_code(self, view):
        view_mapping = {'sag': 0, 'cor': 1, 'tra': 2}
        return view_mapping.get(view, 0)

    def _group_by_patient(self):
        groups = defaultdict(list)
        for key in self._data.keys():
            info = self.case_info[key]
            if info['id'] != 'unknown':
                groups[info['id']].append(key)
        return groups

    def generate_train_batch(self):
        batch = super(UMDDataLoader3D, self).generate_train_batch()

        case_info = []
        for key in batch['keys']:
            case_info.append(self.case_info[key])

        batch['case_info'] = case_info
        return batch


class UMDSequentialDataLoader(UMDDataLoader3D):
    def __init__(self, data, patch_size, final_patch_size, batch_size, has_prev_stage=False,
                 oversample_foreground_percent=0.0, memmap_mode="r", pad_mode="edge",
                 pad_kwargs_data=None, pad_sides=None):
        super(UMDSequentialDataLoader, self).__init__(data, patch_size, final_patch_size, batch_size,
                                                      oversample_foreground_percent, memmap_mode, 1, pad_mode,
                                                      pad_kwargs_data, pad_sides)

        self.sequential_data = self._organize_sequential_data()
        self.current_patient_index = 0

    def _organize_sequential_data(self):
        sequential_data = {}

        for patient_id, cases in self.patient_groups.items():
            if len(cases) >= 2:
                view_order = {'sag': 0, 'cor': 1, 'tra': 2}
                sorted_cases = sorted(cases, key=lambda x: view_order.get(self.case_info[x]['view'], 3))
                sequential_data[patient_id] = sorted_cases

        return sequential_data

    def generate_train_batch(self):
        if not self.sequential_data:
            return super(UMDSequentialDataLoader, self).generate_train_batch()

        patient_ids = list(self.sequential_data.keys())
        current_patient_id = patient_ids[self.current_patient_index]
        patient_cases = self.sequential_data[current_patient_id]

        try:
            batch = self._generate_multiview_batch(patient_cases)

            case_info = []
            for key in batch['keys']:
                case_info.append(self._parse_case_info(key))
            batch['case_info'] = case_info

            if len(batch['keys']) >= 2:
                consistency_pairs = []
                for i in range(len(batch['keys'])):
                    for j in range(i + 1, len(batch['keys'])):
                        consistency_pairs.append((i, j))
                batch['consistency_pairs'] = consistency_pairs
            else:
                batch['consistency_pairs'] = []

            self._update_indices()
            return batch

        except Exception:
            self._update_indices()
            return self.generate_train_batch()

    def _generate_multiview_batch(self, patient_cases):
        all_data = []
        all_seg = []
        all_properties = []
        all_keys = []

        for case_key in patient_cases:
            if case_key not in self._data:
                continue

            case_data = self._data[case_key]
            npy_file = case_data['data_file'][:-4] + ".npy"

            if isfile(npy_file):
                case_all_data = np.load(npy_file, self.memmap_mode)
            else:
                npz_data = np.load(case_data['data_file'])
                case_all_data = npz_data['data']

            num_channels = case_all_data.shape[0]
            if num_channels == 1:
                data = case_all_data[0:1]
                seg = np.zeros((1,) + case_all_data.shape[1:], dtype=np.float32)
            else:
                data = case_all_data[0:1]
                seg = case_all_data[1:]

            data_patch, seg_patch, properties = self._get_patch_from_case(data, seg, case_key)

            all_data.append(data_patch)
            all_seg.append(seg_patch)
            all_properties.append(properties)
            all_keys.append(case_key)

        if not all_data:
            raise ValueError("无法加载任何视角的数据")

        data_shapes = [data.shape for data in all_data]
        seg_shapes = [seg.shape for seg in all_seg]

        if len(set(data_shapes)) > 1:
            raise ValueError(f"数据patch形状不一致: {data_shapes}")
        if len(set(seg_shapes)) > 1:
            raise ValueError(f"标签patch形状不一致: {seg_shapes}")

        batch_data = np.stack(all_data, axis=0)
        batch_seg = np.stack(all_seg, axis=0)

        return {
            'data': batch_data,
            'seg': batch_seg,
            'properties': all_properties,
            'keys': all_keys
        }

    def _get_patch_from_case(self, data, seg, case_key):
        shape = data.shape[1:]
        start = [max(0, (s - p) // 2) for s, p in zip(shape, self.patch_size)]

        need_padding = False
        pad_amounts = [0, 0, 0]

        for i in range(3):
            if start[i] + self.patch_size[i] > shape[i]:
                need_padding = True
                pad_amounts[i] = start[i] + self.patch_size[i] - shape[i]
                start[i] = max(0, shape[i] - self.patch_size[i])

        data_patch = data[:,
                     start[0]:start[0] + self.patch_size[0],
                     start[1]:start[1] + self.patch_size[1],
                     start[2]:start[2] + self.patch_size[2]]

        seg_patch = seg[:,
                    start[0]:start[0] + self.patch_size[0],
                    start[1]:start[1] + self.patch_size[1],
                    start[2]:start[2] + self.patch_size[2]]

        if need_padding:
            data_patch = np.pad(data_patch,
                                ((0, 0), (0, pad_amounts[0]), (0, pad_amounts[1]), (0, pad_amounts[2])),
                                mode='constant', constant_values=0)
            seg_patch = np.pad(seg_patch,
                               ((0, 0), (0, pad_amounts[0]), (0, pad_amounts[1]), (0, pad_amounts[2])),
                               mode='constant', constant_values=0)

        properties = {
            'case_key': case_key,
            'original_shape': shape,
            'patch_size': self.patch_size,
            'start': start,
            'need_padding': need_padding,
            'pad_amounts': pad_amounts
        }

        return data_patch, seg_patch, properties

    def _update_indices(self):
        patient_ids = list(self.sequential_data.keys())
        self.current_patient_index = (self.current_patient_index + 1) % len(patient_ids)


class UMDDataLoader2D(DataLoader2D):
    def __init__(self, data, patch_size, final_patch_size, batch_size, oversample_foreground_percent=0.0,
                 memmap_mode="r", pseudo_3d_slices=1, pad_mode="edge",
                 pad_kwargs_data=None, pad_sides=None):
        super(UMDDataLoader2D, self).__init__(data, patch_size, final_patch_size, batch_size,
                                              oversample_foreground_percent, memmap_mode, pseudo_3d_slices,
                                              pad_mode, pad_kwargs_data, pad_sides)

        self.case_info = {}
        for key in self._data.keys():
            self.case_info[key] = self._parse_case_info(key)

        self.patient_groups = self._group_by_patient()

    def _parse_case_info(self, case_key):
        match = re.match(r'UMD_(\d+)_(sag|cor|tra)', case_key)
        if match:
            return {
                'case_key': case_key,
                'id': match.group(1),
                'view': match.group(2),
                'view_code': self._get_view_code(match.group(2))
            }
        else:
            return {'case_key': case_key, 'id': 'unknown', 'view': 'unknown', 'view_code': 0}

    def _get_view_code(self, view):
        view_mapping = {'sag': 0, 'cor': 1, 'tra': 2}
        return view_mapping.get(view, 0)

    def _group_by_patient(self):
        groups = defaultdict(list)
        for key in self._data.keys():
            info = self.case_info[key]
            if info['id'] != 'unknown':
                groups[info['id']].append(key)
        return groups

    def generate_train_batch(self):
        batch = super(UMDDataLoader2D, self).generate_train_batch()

        case_info = []
        for key in batch['keys']:
            case_info.append(self.case_info[key])

        batch['case_info'] = case_info
        return batch


def load_umd_dataset(folder, num_cases_properties_loading_threshold=1000):
    case_identifiers = get_case_identifiers(folder)
    case_identifiers.sort()
    dataset = OrderedDict()

    for c in case_identifiers:
        dataset[c] = OrderedDict()
        dataset[c]['data_file'] = join(folder, "%s.npz" % c)
        dataset[c]['properties_file'] = join(folder, "%s.pkl" % c)

        match = re.match(r'UMD_(\d+)_(sag|cor|tra)', c)
        if match:
            dataset[c]['patient_id'] = match.group(1)
            dataset[c]['view'] = match.group(2)
            dataset[c]['view_code'] = {'sag': 0, 'cor': 1, 'tra': 2}[match.group(2)]
        else:
            dataset[c]['patient_id'] = 'unknown'
            dataset[c]['view'] = 'unknown'
            dataset[c]['view_code'] = 0

    if len(case_identifiers) <= num_cases_properties_loading_threshold:
        for i in dataset.keys():
            dataset[i]['properties'] = load_pickle(dataset[i]['properties_file'])

    return dataset


def get_case_identifiers(folder):
    return [i[:-4] for i in os.listdir(folder) if i.endswith("npz") and (i.find("segFromPrevStage") == -1)]