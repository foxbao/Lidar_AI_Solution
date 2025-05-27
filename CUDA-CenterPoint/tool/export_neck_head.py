# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# import sys; sys.path.insert(0, "./CenterPoint")

import os
import pickle
import torch
import onnx
import argparse
from onnxsim import simplify
import numpy as np
from torch import nn


from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

import open3d as o3d
import glob
from pathlib import Path


def simplify_model(model_path):
    model = onnx.load(model_path)
    if model is None:
        print("File %s is not find! "%model_path)
    return simplify(model)

def read_pcd(filepath):
    pcd = o3d.io.read_point_cloud(filepath)
    pcd_points = np.asarray(pcd.points)  
    intensities = np.asarray(pcd.colors)[:, 0] if pcd.has_colors() else np.zeros(pcd_points.shape[0])
    point_cloud_data = np.column_stack((pcd_points, intensities))
    point_cloud_data = point_cloud_data.astype(np.float32)
    return  point_cloud_data


class MyKittiDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        elif self.ext == '.pcd':
            pcd_file = self.sample_file_list[index]
            points = read_pcd(str(pcd_file))           
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }


        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

def arg_parser():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', 
                        type=str, default='这里填centerpoint.yaml配置文件',
                        help='specify the config for demo')

    parser.add_argument('--data_path', type=str, 
                        default='这里填一个训练数据集中的点云文件路径(.pcd/.bin/.npy都可以)',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, 
                        default='这里填.pth模型文件', 
                        help='specify the pretrained model')
    # 点云文件后缀(.pcd/.bin/.npy)
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--half', type=bool, default=False, help='True:export FP16 onnx model, else, FP32 model')
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1]) 
    return args, cfg

class CenterPointVoxelNet_Post(nn.Module):
    def __init__(self, model):
        super(CenterPointVoxelNet_Post, self).__init__()
        self.model = model
        
        # assert( len(self.model.bbox_head.tasks) == 6 )
        assert( len(self.model.dense_head.heads_list) == 1 )
        

    def forward(self, x):
        data_dict = {}
        data_dict['spatial_features'] = x
        x = self.model.backbone_2d(data_dict)
        x = data_dict['spatial_features_2d']
        x = self.model.dense_head.shared_conv(x)
        pred = [ task(x) for task in self.model.dense_head.heads_list ]
        
        return pred[0]['center'], pred[0]['center_z'], pred[0]['dim'], pred[0]['rot'], pred[0]['hm']

def main():
    args, cfg = arg_parser()
    logger = common_utils.create_logger()
    logger.info(' *************** CenterPoint Export NeckHead Onnx Model *****************')
    if args.data_path.endswith(".bin") or args.data_path.endswith(".pcd") or args.data_path.endswith(".npy"):
        # 数据加载
        demo_dataset = MyKittiDataset(
            dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
            root_path=Path(args.data_path), ext=args.ext, logger=logger
        )
        # 模型加载
        # neck + head = mep_to_bev + backbone2d + ceneterhead
        model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
        model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
        post_model = CenterPointVoxelNet_Post(model)

        if args.half:
            model.eval().half()
            post_model.eval().half()
        else:
            model.eval()
            post_model.eval()

        # model to cuda
        model = model.cuda()
        post_model = post_model.cuda()
        with torch.no_grad():
            # ====== 3d backbone output( add maptobev)
            # x = ret (shape = (N,C*D,H,W))
            # 这里的neck_head_input_shape可以通过运行下面的命令得到(需要在pcdet/models/backbones_2d/map_to_bev/height_compression.py里新增一行打印)
            '''
             python tools/demo.py --cfg_file centerpoint.yaml  --ckpt centerpoint.pth  --data_path 数据集中的某一个数据文件(.bin或者.pcd或者.npy) 
            '''
            # 输出：======================:N, C*D, H, w =(), 即为neck_head_input_shape的值
            # neck_head_input_shape = (1,256,82,94)
            
            # neck_head_input_shape=[(70.4)/0.05/8, (40-(-40))/0.05/8]
            neck_head_input_shape = (1,256,200,176)

            rpn_input  = torch.zeros(neck_head_input_shape,dtype=torch.float32,device=torch.device("cuda"))
            if args.half:
                rpn_input  = rpn_input.half()
            # ===== export_params 将模型的参数（权重+偏置）导出到onnx文件
            # ===== pcdet:传参仍然为dict， post_model: rpn_input 
            torch.onnx.export(post_model, rpn_input, "pcdet_neck_head.onnx",
            export_params=True, opset_version=11, do_constant_folding=True,
            keep_initializers_as_inputs=False, input_names = ['input'],
            output_names = ['reg_0', 'height_0', 'dim_0', 'rot_0','hm_0'],
            )
            sim_model, check = simplify_model("pcdet_neck_head.onnx")
            if not check:
                print("[ERROR]:Simplify %s error!"% "tmp.onnx")
            onnx.save(sim_model, "pcdet_neck_head_sim.onnx")
            print("[PASS] Export ONNX done.")
    logger.info('************ export onnx Model complete... *************')



if __name__ == "__main__":
    # args = arg_parser()
    main()