from torch.utils.data import Dataset
from matplotlib import pyplot as plt
from utils.CSL_data_preprocessing import CSL_data_preprocessing as utils
import numpy as np
import random

class CSL_dataset(Dataset):
    """
    Initializes the dataset class.

    Args:
        root (tuple): Tuple containing data, label, and num_classes.
        width (int): Width of the data.
        height (int): Height of the data.
        train (bool): Specifies if the dataset is for training or testing.
        channel (int): Number of channels in the data.
        fold (int): Number of folds for data splitting.
        fold_order (int): Order of the fold to be used.

    Raises:
        AssertionError: If fold_order is greater than or equal to fold.
    """
    def __init__(self, root,width=8,height=24,train=False,channel=1,fold=2,fold_order=1):
        
        assert fold_order < fold, "fold_order has to be less than fold"
        self.channel = channel
        self.width = width
        self.height = height
        self.utils = utils()
        self.data, self.label, self.num_classes = root[0],root[1],root[2]
        
        # Shuffle the data and label indexes
        data_indexes = self.data.shape[0] #self.data.shape = (1228800, 192)
        random.seed(42)
        data_indexes_list = [i for i in range(data_indexes)]
        random.shuffle(data_indexes_list)
        self.data = self.data[data_indexes_list]
        self.label = self.label[data_indexes_list]
        
        # # Split the data and label based on the fold_order
        fold_nodes = [i*(data_indexes//(fold+1)) for i in range(fold+1)]
        if train:
            # Concatenate the data and label from the specified fold_order
            self.data = np.concatenate(
                (self.data[fold_nodes[0]:fold_nodes[fold_order]],self.data[fold_nodes[fold_order+1]:]))
            self.label = np.concatenate(
                (self.label[fold_nodes[0]:fold_nodes[fold_order]], self.label[fold_nodes[fold_order+1]:]))
        else:
            self.data = self.data[fold_nodes[fold_order]:fold_nodes[fold_order+1]]
            self.label = self.label[fold_nodes[fold_order]:fold_nodes[fold_order+1]]

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.data)

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
        data = self.utils.NormalizeData(data)

        
        data = data.reshape(self.channel,self.width,self.height)

        data = data.astype("float32")
        return data,label
