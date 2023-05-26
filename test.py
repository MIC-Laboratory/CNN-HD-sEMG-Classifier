import torch
import scipy.io
from models.mobilenetv2 import MobileNetV2

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
model = MobileNetV2(num_classes=8, input_layer=1, model_width=1)

# Load the pre-trained model weights
model.load_state_dict(torch.load("pretrain_model/MobilenetV2_Params@2.314M_MAC@18.303M_Acc@99.197.pt"), strict=False)

# Set the model to evaluation mode
model.eval()

# Load input data from a MATLAB file
input_data = scipy.io.loadmat("data/sEMG_Test_Label_hand_close.mat")['Data']

# Convert the input data to a PyTorch tensor
input_data = torch.asarray(input_data)

# Reshape the input data to match the expected shape of the model
input_data = input_data.reshape(-1, 1, 8, 24)

# Normalize the input data by subtracting the mean and dividing by the standard deviation
input_data = (input_data - input_data.mean()) / input_data.std()

# Disable gradient calculation since we're only doing inference
with torch.no_grad():
    print(f"Number of Input sEMG samples:{input_data.shape[0]}, Labels for those samples are Hand Close")
    # Pass the input data through the model
    outputs = model(input_data)
    # Count the number of occurrences of the most frequent output class
    output_count = (outputs.topk(1).indices.reshape(-1) == 0).sum(dim=0).item()

    # Get the total number of outputs
    total = input_data.shape[0]

    # Print the most frequent output class and accuracy
    print(f"{output_count} of samples predict correct")
    print(f"Accuracy: {output_count / total:.3f}")
