import scipy.io
import numpy as np
from tqdm import tqdm

class ICE_lab_data_preprocessing:
    
    def load_from_mat(self,fileAddress):
        """
        Load data from a MATLAB file.

        Args:
            fileAddress (str): The address of the MATLAB file.

        Returns:
            numpy.ndarray: The data stored under the key 'Data' in the MATLAB file.
        """
        return scipy.io.loadmat(fileAddress)['Data']

    def extra_data(self,fileAddress):
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
        num_class = 8
        for gest in tqdm(range(1,num_class+1),desc="Processing Files"):
            for i in tqdm(range(1,5),desc="Processing Files"):
                if data is None:
                    data = self.load_from_mat(f"{fileAddress}/001-00{gest}-00{i}.mat")
                    label = np.repeat(gest-1,data.shape[0])
                else:
                    temp = self.load_from_mat(f"{fileAddress}/001-00{gest}-00{i}.mat")
                    if temp.shape[1] == 193:
                        temp = temp[:,:-1]
                    data = np.concatenate((data,temp),axis=0)
                    label = np.concatenate((label,np.repeat(gest-1,temp.shape[0])),axis=0)
        return data,label,num_class

    def NormalizeData(self,data):
        data = (data - data.mean())/(data.std())
        return data