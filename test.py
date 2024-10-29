import torch
import scipy.io
import numpy as np
from models.M5 import M5


    
def concatenate_rows_in_sliding_window(matrix, timeWindow=20):
    
    """
    This function processes a given matrix by concatenating a specified number of consecutive rows 
    into a single row in a sliding window fashion. The function expects the input matrix to have 
    exactly 192 columns. It slides over the matrix row by row, taking 'rows_to_concat' consecutive 
    rows each time, concatenates them horizontally, and stores the result in a new matrix.

    Parameters:
    matrix (numpy.ndarray): The input matrix with shape (n_rows, 192).
    rows_to_concat (int): The number of consecutive rows to concatenate. Default is 20.

    Returns:
    numpy.ndarray: A new matrix where each row is the concatenation of 'rows_to_concat' consecutive 
                    rows from the input matrix.
    """
    
    # Check the matrix shape
    n_rows, n_cols = matrix.shape
    if n_cols != 192:
        raise ValueError(f"Expected 192 columns, but got {n_cols}.")

    # Prepare the new data structure
    new_matrix_list = []

    # Iterate through the matrix in a sliding window fashion
    for i in range(n_rows - timeWindow + 1):
        # Take 20 consecutive rows and concatenate them horizontally
        concatenated = matrix[i:i + timeWindow, :].reshape(-1)
        new_matrix_list.append(concatenated)

    # Convert the list to a NumPy array
    new_matrix = np.array(new_matrix_list)
    return new_matrix





classes = {
    0:"Hand Close",
    1:"No Movement",
    2:"Hand Open",
    3:"Wrist Supination",
    4:"Wrist Pronation",
    5:"Wrist Extension",
    6:"Wrist Flexion",
}
# Create an instance of the M5 model with specified parameters
model = M5(num_classes=7)

# Load the pre-trained model weights
model.load_state_dict(torch.load(
    "pretrain_model/M5_recover.pt"))

# Set the model to evaluation mode
model.eval()

for index,label in classes.items():
    
    # Load input data from a MATLAB file
    input_data = scipy.io.loadmat(f"data/001-00{index+1}-008.mat")['data']

    # Remove unusual data
    if input_data.shape[1] == 193:
        input_data = input_data[:,:-1]
    # Reshape the input data to match the expected shape of the model
    input_data = concatenate_rows_in_sliding_window(input_data)
    # Convert the input data to a PyTorch tensor
    input_data = torch.asarray(input_data)
    input_data = input_data.float()
    input_label = index
    input_label = torch.asarray(input_label)
    # Disable gradient calculation since we're only doing inference
    with torch.no_grad():
        print(f"Number of Input sEMG samples:{input_data.shape[0]}, Labels for those samples are {label}")
        # Pass the input data through the model
        outputs = model(input_data)
        # Count the number of occurrences of the most frequent output class
        output_count = (outputs.topk(1).indices.reshape(-1) == input_label).sum(dim=0).item()
        # Get the total number of outputs
        total = input_data.shape[0]

        # Print the most frequent output class and accuracy
        print(f"{output_count} of samples predict correct")
        print(f"Accuracy: {output_count / total:.3f}")
