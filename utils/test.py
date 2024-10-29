import scipy.io as sio
import numpy as np
import os

def process_mat_file(input_filepath, output_filepath, rows_to_concat=20):
    # Load the .mat file
    data = sio.loadmat(input_filepath)

    # Assuming the matrix is stored under a variable called 'data'
    matrix = data['data']

    # Check the matrix shape
    n_rows, n_cols = matrix.shape
    if n_cols != 192:
        raise ValueError(f"Expected 192 columns, but got {n_cols}.")

    # Prepare the new data structure
    new_matrix_list = []

    # Iterate through the matrix in a sliding window fashion
    for i in range(n_rows - rows_to_concat + 1):
        # Take 20 consecutive rows and concatenate them horizontally
        concatenated = matrix[i:i + rows_to_concat, :].reshape(-1)
        new_matrix_list.append(concatenated)

    # Convert the list to a NumPy array
    new_matrix = np.array(new_matrix_list)

    # Save the new matrix with 3840 columns (20 rows concatenated with 192 columns)
    sio.savemat(output_filepath, {'data': new_matrix})

def process_all_mat_files(input_folder, output_folder, rows_to_concat=20):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all .mat files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".mat"):
            input_filepath = os.path.join(input_folder, filename)
            output_filepath = os.path.join(output_folder, filename)
            
            # Process each .mat file
            print(f"Processing {filename}...")
            process_mat_file(input_filepath, output_filepath, rows_to_concat)

    print(f"All .mat files processed and saved to {output_folder}.")

# Example usage:
input_folder = './output_mat_files'
output_folder = './sliding_window_mat_files'
process_all_mat_files(input_folder, output_folder)