import scipy.io
import numpy as np
from tqdm import tqdm

class ICE_lab_data_preprocessing:
    
    
    
    def concatenate_rows_in_sliding_window(self,matrix, timeWindow=20):
        
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
    
    def load_from_mat(self,fileAddress):
        """
        Load data from a MATLAB file.

        Args:
            fileAddress (str): The address of the MATLAB file.

        Returns:
            numpy.ndarray: The data stored under the key 'Data' in the MATLAB file.
        """
        return scipy.io.loadmat(fileAddress)['data']

    def extra_data(self,fileAddress, timeWindow=20):
        """
        Load and concatenate data from multiple MATLAB files, generating corresponding labels.

        Args:
            fileAddress (str): The address of the directory containing the MATLAB files.

        Returns:
            tuple: A tuple containing two numpy.ndarrays. The first element is the concatenated data,
                   and the second element is the corresponding labels. The third element is the number of classes
        """
        data = None
        label = None
        num_class = 7  # Assuming there are 7 classes
        for gest in tqdm(range(1,num_class + 1,1),desc="Processing Files"):
            for i in tqdm(range(1,8),desc="Processing Files"):
                if data is None:
                    data = self.load_from_mat(f"{fileAddress}/001-00{gest}-00{i}.mat")
                    
                    if data.shape[1] == 193:
                        data = data[:,:-1]
                    data = self.concatenate_rows_in_sliding_window(data,timeWindow)
                    label = np.repeat(gest-1,data.shape[0])
                    
                else:
                    temp = self.load_from_mat(f"{fileAddress}/001-00{gest}-00{i}.mat")
                    if temp.shape[1] == 193:
                        temp = temp[:,:-1]
                    temp = self.concatenate_rows_in_sliding_window(temp)
                    data = np.concatenate((data,temp),axis=0)
                    label = np.concatenate((label,np.repeat(gest-1,temp.shape[0])),axis=0)
        return data,label,num_class

    def NormalizeData(self,data):
        data = (data - data.mean())/(data.std())
        return data