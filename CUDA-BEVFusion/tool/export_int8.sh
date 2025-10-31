#!/bin/bash
set -e  # 如果有一条命令出错，就终止脚本
# 导出自己训练的模型
python qat/export-camera.py --ckpt=qat/ckpt/bevfusion_ptq.pth
python qat/export-transfuser.py --ckpt=qat/ckpt/bevfusion_ptq.pth
python qat/export-camera.py --ckpt=qat/ckpt/bevfusion_ptq.pth