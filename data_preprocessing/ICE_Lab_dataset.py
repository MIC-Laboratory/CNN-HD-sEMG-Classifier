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
        width (int): Width of the data.
        height (int): Height of the data.
        train (bool): Flag indicating whether to use training or testing data.
        channel (int): Number of channels in the data.
        fold (int): Number of data folds for splitting the dataset.
        fold_order (int): The order of the fold to be used.

    Attributes:
        root (str): Root directory of the dataset.
        channel (int): Number of channels in the data.
        width (int): Width of the data.
        height (int): Height of the data.
        utils (object): An instance of the utils class for data processing.
        data (numpy.ndarray): The loaded data from the dataset.
        label (numpy.ndarray): The corresponding labels for the data.
        num_classes (int): The number of classes in the dataset.
        
        Raises:
        AssertionError: If fold_order is greater than or equal to fold.
    """
    def __init__(self, root,width=8,height=24,train=False,channel=1):
        
        self.root = root
        self.channel = channel
        self.width = width
        self.height = height
        self.utils = utils()
        self.data, self.label, self.num_classes = root[0],root[1],root[2]
        
        # # Shuffle the data and label indexes
        X_train, X_test, y_train, y_test = train_test_split(self.data, self.label, test_size=0.33, random_state=42)
        if train:
            self.data = X_train
            self.label = y_train
        else:
            self.data = X_test
            self.label = y_test

        

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
        data = data.reshape(self.channel,self.width,self.height)
        data = data.astype("float32")
        
        
        return data,label
