import torch
import scipy.io
import numpy as np
from models.mobilenetv1 import MobilenetV1

classes = {
    0:"Hand Close",
    1:"No Movement",
    2:"Hand Open",
    3:"Wrist Supination",
    4:"Wrist Pronation",
    5:"Wrist Extension",
    6:"Wrist Flexion",
    7:"N/A"
}
# Create an instance of the MobileNetV2 model with specified parameters
model = MobilenetV1(ch_in=1,n_classes=8,Global_ratio=1)

# Load the pre-trained model weights
model.load_state_dict(torch.load(
    "pretrain_model/MobilenetV1_Param@3.21 M  _MAC@5.37 MMac_Acc@95.070.pt"))

# Set the model to evaluation mode
model.eval()

# Load input data from a MATLAB file
input_data = scipy.io.loadmat("data/sEMG_Test_Label_hand_close.mat")['Data']
# Convert the input data to a PyTorch tensor
input_data = torch.asarray(input_data)
# Remove unusual data
if input_data.shape[1] == 193:
    input_data = input_data[:,:-1]
# Reshape the input data to match the expected shape of the model
input_data = input_data.reshape(-1, 1, 8, 24)


input_label = 0
input_label = torch.asarray(input_label)
# Disable gradient calculation since we're only doing inference
with torch.no_grad():
    print(f"Number of Input sEMG samples:{input_data.shape[0]}, Labels for those samples are Hand Close")
    # Pass the input data through the model
    outputs = model(input_data)
    # Count the number of occurrences of the most frequent output class
    output_count = (outputs.topk(1).indices.reshape(-1) == input_label).sum(dim=0).item()
    # print(outputs.topk(1).indices.reshape(-1))
    # Get the total number of outputs
    total = input_data.shape[0]

    # Print the most frequent output class and accuracy
    print(f"{output_count} of samples predict correct")
    print(f"Accuracy: {output_count / total:.3f}")
