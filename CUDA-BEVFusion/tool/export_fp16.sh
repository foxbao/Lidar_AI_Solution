#!/bin/bash
set -e  # 如果有一条命令出错，就终止脚本
# 导出自己训练的模型
python qat/export-camera.py --ckpt=qat/ckpt/bevfusion_ptq.pth --fp16
python qat/export-transfuser.py --ckpt=qat/ckpt/bevfusion_ptq.pth --fp16
python qat/export-scn.py --ckpt=qat/ckpt/bevfusion_ptq.pth --save=qat/onnx_fp16/lidar.backbone.onnx