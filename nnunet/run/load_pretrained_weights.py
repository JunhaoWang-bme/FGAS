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
import torch

def load_pretrained_weights(network, fname, verbose=False):
    """
    THIS DOES NOT TRANSFER SEGMENTATION HEADS!
    """
    saved_model = torch.load(fname, weights_only=False)
    pretrained_dict = saved_model['state_dict']

    new_state_dict = {}

    # if state dict comes from nn.DataParallel but we use non-parallel model here then the state dict keys do not
    # match. Use heuristic to make it match
    for k, value in pretrained_dict.items():
        key = k
        # remove module. prefix from DDP models
        if key.startswith('module.'):
            key = key[7:]
        new_state_dict[key] = value

    pretrained_dict = new_state_dict

    model_dict = network.state_dict()
    ok = True
    for key, _ in model_dict.items():
        if ('conv_blocks' in key):
            if (key in pretrained_dict) and (model_dict[key].shape == pretrained_dict[key].shape):
                continue
            else:
                ok = False
                break

    # filter unnecessary keys
    if ok:
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                           (k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape)}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        print("################### Loading pretrained weights from file ", fname, '###################')
        if verbose:
            print("Below is the list of overlapping blocks in pretrained model and nnUNet architecture:")
            for key, _ in pretrained_dict.items():
                print(key)
        print("################### Done ###################")
        network.load_state_dict(model_dict)
    else:
        raise RuntimeError("Pretrained weights are not compatible with the current network architecture")


# import torch
#
# import torch
#
#
# def load_pretrained_weights(network, fname, verbose=True, freeze_loaded_layers=True):
#     saved_model = torch.load(fname, weights_only=False)
#     pretrained_dict = saved_model['state_dict']
#
#     # 处理DDP模型的module.前缀
#     new_state_dict = {}
#     for k, value in pretrained_dict.items():
#         key = k
#         if key.startswith('module.'):
#             key = key[7:]
#         new_state_dict[key] = value
#     pretrained_dict = new_state_dict
#
#     model_dict = network.state_dict()
#     # 定义要尝试加载的目标层：上下文块前3层（放宽匹配条件，只要包含前缀即可）
#     target_layer_prefixes = ['conv_blocks_context.0.', 'conv_blocks_context.1.', 'conv_blocks_context.2.']
#     target_keys = [k for k in model_dict.keys() if any(prefix in k for prefix in target_layer_prefixes)]
#
#     # ########## 关键：打印所有目标层的维度对比 ##########
#     print("\n=== 上下文块前3层维度对比（当前网络 vs 预训练权重）===")
#     compatible_keys = []  # 记录维度匹配的层
#     incompatible_keys = []  # 记录不匹配的层
#     for key in target_keys:
#         current_shape = model_dict[key].shape
#         pretrained_shape = pretrained_dict.get(key, "不存在")
#         if key in pretrained_dict and current_shape == pretrained_shape:
#             compatible_keys.append(key)
#             print(f"✅ 匹配: {key} | 当前形状: {current_shape} | 预训练形状: {pretrained_shape}")
#         else:
#             incompatible_keys.append(key)
#             print(f"❌ 不匹配: {key} | 当前形状: {current_shape} | 预训练形状: {pretrained_shape}")
#     print("==================================================\n")
#
#     # 仅加载维度匹配的层（即使只有部分层匹配也加载，不强制要求所有3层都兼容）
#     if len(compatible_keys) > 0:
#         pretrained_dict_filtered = {k: pretrained_dict[k] for k in compatible_keys}
#         # 更新网络权重
#         model_dict.update(pretrained_dict_filtered)
#         network.load_state_dict(model_dict)
#
#         # 打印加载结果
#         print(f"################### 加载成功 ###################")
#         print(f"从 {fname} 加载了 {len(compatible_keys)} 个匹配的层：")
#         for k in compatible_keys:
#             print(f"- {k}")
#         print("################################################\n")
#
#         # 冻结已加载的层
#         if freeze_loaded_layers:
#             print("冻结已加载的层，不参与训练更新")
#             for name, param in network.named_parameters():
#                 if any(k in name for k in compatible_keys):
#                     param.requires_grad = False
#     else:
#         # 若完全没有匹配的层，打印更详细的提示
#         raise RuntimeError(
#             f"预训练权重与当前网络的上下文块前3层无任何兼容层！\n"
#             f"当前网络前3层用 (1,3,3) 卷积核（适配轴向分辨率5.5mm），\n"
#             f"预训练权重前3层用 (3,3,3) 卷积核（适配各向同性图像），\n"
#             f"建议：1）尝试加载前2层/前1层；2）不使用该权重从零训练。"
#         )