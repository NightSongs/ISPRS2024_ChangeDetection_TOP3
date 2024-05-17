import os

import torch
from torch.optim.swa_utils import AveragedModel
from nets.upernet import UPerNet

model = UPerNet(backbone_name='convnext_large_clip',
                dropout=0.5,
                drop_path_rate=0.5,
                pretrained=True,
                num_classes=7,
                fusion_form='conv',
                scse=False)

model_name = "upernet_convnext_large_clip_drop0.5_droppath0.5_Convf_62e_aug"

torch_model_path = f"/mnt/d/competition/ISPRS2024/code/checkpoints/{model_name}/seg_model_1.pth"
onnx_model_path = f"/mnt/d/competition/ISPRS2024/code/checkpoints/onnx/{model_name}"
os.makedirs(onnx_model_path, exist_ok=True)
onnx_model_path = os.path.join(onnx_model_path, "seg_model_1.onnx")

model.load_state_dict(torch.load(torch_model_path))

model.eval()

# 随机生成的输入参数(shape需要和模型输入对应)
x1 = torch.randn((1, 3, 512, 512))
x2 = torch.randn((1, 3, 512, 512))

input_names = ["x1", "x2"]

torch.onnx.export(model=model,
                  args=(x1, x2),
                  f=onnx_model_path,  # 转换输出的模型的地址
                  input_names=input_names,  # 指定输入节点名称
                  opset_version=11,  # 默认的9不支持Upsample/Resize
                  )
