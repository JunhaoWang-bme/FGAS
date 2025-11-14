#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import sys
import shutil
import re

# 添加nnU-Net路径
current_dir = os.path.dirname(os.path.abspath(__file__))
nnunet_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(nnunet_root)

from collections import OrderedDict
from nnunet.paths import nnUNet_raw_data
from batchgenerators.utilities.file_and_folder_operations import *
import shutil
import os
import re


if __name__ == "__main__":
    # 配置参数
    base = "/path/to/your/UMD/data/"  # 请修改为您的数据路径
    task_id = 999  # 请修改为您的任务ID
    task_name = "UMD_MultiView"
    
    # 视图映射
    view_mapping = {
        'sag': '0000',  # 矢状面
        'cor': '0001',  # 冠状面  
        'tra': '0002'   # 横断面
    }
    
    foldername = "Task%03.0d_%s" % (task_id, task_name)
    
    out_base = join(nnUNet_raw_data, foldername)
    imagestr = join(out_base, "imagesTr")
    imagests = join(out_base, "imagesTs")
    labelstr = join(out_base, "labelsTr")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)
    
    # 获取所有图像文件
    image_files = []
    label_files = []
    
    # 扫描数据目录
    for root, dirs, files in os.walk(base):
        for file in files:
            if file.endswith('.nii.gz'):
                file_path = join(root, file)
                # 检查是否为图像文件（包含_0000.nii.gz）
                if '_0000.nii.gz' in file:
                    image_files.append(file_path)
                # 检查是否为标签文件（不包含_0000.nii.gz）
                elif not any(f'_000{i}.nii.gz' in file for i in range(10)):
                    label_files.append(file_path)
    
    print(f"Found {len(image_files)} image files")
    print(f"Found {len(label_files)} label files")
    
    # 处理训练数据
    train_patient_names = []
    processed_pairs = set()
    
    for img_file in image_files:
        # 解析文件名 UMD_id_view_0000.nii.gz
        filename = os.path.basename(img_file)
        match = re.match(r'UMD_(\d+)_(sag|cor|tra)_0000\.nii\.gz', filename)
        
        if match:
            patient_id = match.group(1)
            view = match.group(2)
            
            # 构建对应的标签文件名
            label_filename = f'UMD_{patient_id}_{view}.nii.gz'
            label_file = None
            
            # 查找对应的标签文件
            for lbl_file in label_files:
                if os.path.basename(lbl_file) == label_filename:
                    label_file = lbl_file
                    break
            
            if label_file is not None:
                # 创建nnU-Net格式的文件名
                # 图像文件：保持原格式 UMD_id_view_0000.nii.gz
                nnunet_image_name = f'UMD_{patient_id}_{view}_0000.nii.gz'
                # 标签文件：去掉_0000后缀 UMD_id_view.nii.gz
                nnunet_label_name = f'UMD_{patient_id}_{view}.nii.gz'
                
                # 复制文件到nnU-Net标准目录结构
                shutil.copy(img_file, join(imagestr, nnunet_image_name))
                shutil.copy(label_file, join(labelstr, nnunet_label_name))
                
                # 记录处理的患者
                patient_key = f'UMD_{patient_id}_{view}'
                if patient_key not in processed_pairs:
                    train_patient_names.append(patient_key)
                    processed_pairs.add(patient_key)
                
                print(f"Processed: {filename} -> {nnunet_image_name}")
                print(f"         : {label_filename} -> {nnunet_label_name}")
    
    # 处理测试数据（如果有的话）
    test_patient_names = []
    # 这里可以添加测试数据的处理逻辑
    
    # 创建dataset.json
    json_dict = OrderedDict()
    json_dict['name'] = "UMD_MultiView"
    json_dict['description'] = "UMD Multi-View Medical Image Segmentation Dataset"
    json_dict['tensorImageSize'] = "3D"
    json_dict['reference'] = "UMD Dataset"
    json_dict['licence'] = "see dataset license"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "MRI",  # 根据您的实际模态修改
    }
    json_dict['labels'] = OrderedDict({
        "0": "background",
        "1": "target_structure",  # 根据您的实际标签修改
    })
    
    # 添加视图信息
    json_dict['views'] = {
        "0000": "sagittal",
        "0001": "coronal", 
        "0002": "transverse"
    }
    
    json_dict['numTraining'] = len(train_patient_names)
    json_dict['numTest'] = len(test_patient_names)
    json_dict['training'] = [
        {'image': f"./imagesTr/{name}.nii.gz", "label": f"./labelsTr/{name}.nii.gz"} 
        for name in train_patient_names
    ]
    json_dict['test'] = [f"./imagesTs/{name}.nii.gz" for name in test_patient_names]
    
    save_json(json_dict, os.path.join(out_base, "dataset.json"))
    
    print(f"\nDataset conversion completed!")
    print(f"Task folder: {out_base}")
    print(f"Training cases: {len(train_patient_names)}")
    print(f"Test cases: {len(test_patient_names)}")
    print(f"Views supported: {list(view_mapping.keys())}")
    
    # 显示目录结构
    print(f"\nDirectory structure:")
    print(f"{out_base}/")
    print(f"├── imagesTr/")
    print(f"│   ├── UMD_001_sag_0000.nii.gz")
    print(f"│   ├── UMD_001_cor_0000.nii.gz")
    print(f"│   ├── UMD_001_tra_0000.nii.gz")
    print(f"│   └── ...")
    print(f"├── labelsTr/")
    print(f"│   ├── UMD_001_sag.nii.gz")
    print(f"│   ├── UMD_001_cor.nii.gz")
    print(f"│   ├── UMD_001_tra.nii.gz")
    print(f"│   └── ...")
    print(f"├── imagesTs/")
    print(f"└── dataset.json")
