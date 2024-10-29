import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.M5 import M5
import os

model = M5(num_classes=7)
model.load_state_dict(torch.load("pretrain_model/M5_recover.pt"))
model.eval()
dummy_input = torch.randn(1, 3840)  # one sample, 10 features (same as the model's input size)
save_path = "pretrain_model/onnx_model"
if (os.path.isdir(save_path) == False):
    os.makedirs(save_path)
onnx_file = os.path.join(save_path,"M5_recover.onnx") 
torch.onnx.export(model, dummy_input, onnx_file, verbose=True, input_names=['input'], output_names=['output'])