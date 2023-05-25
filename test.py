import torch
from models.mobilenetv2 import MobileNetV2

model = MobileNetV2(
    num_classes=8,input_layer=1,model_width=1)
model.load_state_dict(torch.load("pretrain_model/MobilenetV2_Params@2.314M_MAC@18.303M_Acc@99.026.pt"),strict=False)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# model.to(device)
model.eval()

classes = [
    "No movement",
    "wrist supination",
    "wrist pronation",
    "hand close",
    "hand open",
    "wrist flxion",
    "wrist extension",
    "N/A"
]
with open("testdata_label_1.txt",'r') as f:
    sEMG = f.read()
    sEMG = sEMG.split("\t")
    sEMG = [float(signal) for signal in sEMG]
    sEMG = torch.asarray(sEMG)
sEMG = sEMG.reshape(1,1,8,24)
with torch.no_grad():
    outputs = model(sEMG)  
_, predicted = torch.max(outputs, 1)

print(f"Prediction Gesture:{classes[predicted]},probability:{outputs[predicted]}")
