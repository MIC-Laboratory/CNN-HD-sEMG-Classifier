from torch.utils.data import Dataset
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from utils.ICE_lab_data_preprocessing import ICE_lab_data_preprocessing as utils
import numpy as np
import random

class ICE_Lab_dataset(Dataset):
    """
    Initialize the ICE_Lab_dataset class.

    Args:
        root (tuple): Tuple containing data, label, and num_classes.
        num_sensor (int): Number of sensors: 192.

        channel (int): Number of channels in the data.

    Attributes:
        root (str): Root directory of the dataset.
        channel (int): Number of channels in the data.
        num_sensor (int): Number of sensors: 192.
        utils (object): An instance of the utils class for data processing.
        data (numpy.ndarray): The loaded data from the dataset.
        label (numpy.ndarray): The corresponding labels for the data.
        num_classes (int): The number of classes in the dataset.
    """
    def __init__(self, data,num_sensor,channel=1):
        
        self.data = data
        self.channel = channel
        self.num_sensor = num_sensor
        self.utils = utils()
        self.data, self.label, self.num_classes = data[0],data[1],data[2]

        

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return self.data.shape[0]

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: A tuple containing the data and its corresponding label.
        """
        data = self.data[idx]
        label = self.label[idx].astype(int)
        data = data.reshape(self.channel,self.num_sensor,-1)
        data = data.astype("float32")
        
        
        return data,label
