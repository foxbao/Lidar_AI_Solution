import os
import pickle
import torch
import onnx
import onnxruntime as ort
import argparse
from onnxsim import simplify
import numpy as np
from torch import nn

# from det3d.datasets import build_dataloader, build_dataset
# from det3d.models import build_detector
# from det3d.core import box_torch_ops
# from det3d.torchie import Config
# from det3d.torchie.apis.train import example_to_device
# from det3d.torchie.trainer import load_checkpoint


from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

import open3d as o3d
import glob
from pathlib import Path


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
    parser.add_argument('--scn_onnx_path',type=str, default='centerpoint_pre.scn.onnx', help='specify the onnx model path')
    parser.add_argument('--neck_head_sim_path',default='pcdet_neck_head_sim.onnx', help='specify the onnx model path')
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1]) 
    return args, cfg

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
        
    from pcdet.datasets import build_dataloader
    test_dataset, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=4,
        logger=logger,
        training=False,
    )
    # 模型加载
    # neck + head = mep_to_bev + backbone2d + ceneterhead
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
        
    model.cuda()
    model.eval()
    
    real_input=dict()
    from pcdet.models import load_data_to_gpu
    for i, batch_dict in enumerate(test_dataset):
        data_dict = test_dataset.collate_batch([batch_dict])
        load_data_to_gpu(data_dict)
        real_input["voxels"]=data_dict["voxels"]
        real_input["voxel_num_points"]=data_dict["voxel_num_points"].to(torch.int32)
        real_input["voxel_coords"] = data_dict["voxel_coords"].to(torch.int32)
        real_input["batch_size"] = data_dict["batch_size"]
        data_input=real_input
        break
    
    MeanVFE_part=model.module_list[0]
    VoxelResBackBone8x_part=model.module_list[1]
    HeighCompression_part=model.module_list[2]
    BaseBEVBackbone_part=model.module_list[3]
    CenterHead_part=model.module_list[4]
    

    np.set_printoptions(threshold=np.inf)
    
    print("neck_head!")
    onnx_neck_head_sim=onnx.load(args.neck_head_sim_path)
    ort_session = ort.InferenceSession(
        os.path.join(args.neck_head_sim_path)
    )
    # 检查输入输出是否确实支持动态维度
    for input_tensor in ort_session.get_inputs():
        print(f"Input: {input_tensor.name}, Shape: {input_tensor.shape},Type: {input_tensor.type}")
        # 应该看到类似 image_input: ['batch_size', 3, 'height', 'width'] 的输出

    for output_tensor in ort_session.get_outputs():
        print(f"Output: {output_tensor.name}, Shape: {output_tensor.shape},Type: {output_tensor.type}")
    
    
    
    # test neck_head.onnx
    pass

if __name__ == "__main__":
    # args = arg_parser()
    main()