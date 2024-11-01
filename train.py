

from torch.utils.data import DataLoader
from tqdm import tqdm
from data_preprocessing import ICE_Lab_dataset
from sklearn.model_selection import train_test_split
from models.M5 import M5
from ptflops import get_model_complexity_info
import torch
import torch.nn as nn
import os
import json

# Load configuration from JSON file
config = json.load(open('config.json', 'r'))

# Check the value of the 'dataset' key in the 'config' dictionary

if config["dataset"] == "CSL":
    # Import the necessary modules for CSL dataset
    from data_preprocessing.CSL_dataset import CSL_dataset as dataset
    from utils.CSL_data_preprocessing import CSL_data_preprocessing as utils

    # Extract additional data, labels, and number of classes using CSL data preprocessing utility
    data,label,num_classes = utils().extra_data(config["data_path"])

    # Assign the extracted data, labels, and number of classes to 'root' for further usage

    root = (data, label, num_classes)
elif config["dataset"] == "ICE":
    # Import the necessary modules for ICE dataset
    from data_preprocessing.ICE_Lab_dataset import ICE_Lab_dataset as dataset
    from utils.ICE_lab_data_preprocessing import ICE_lab_data_preprocessing as utils

    # Extract additional data, labels, and number of classes using ICE data preprocessing utility
    data,label,num_classes = utils().extra_data(config["data_path"],config["input_time_window"])
    X_train, X_test, y_train, y_test = train_test_split(data,label, test_size=0.33, random_state=42)

    # Assign the data path from 'config' to 'root' for further usage
    training_data = (X_train, y_train, num_classes)
    testing_data = (X_test, y_test, num_classes)
else:

    # Raise an exception if the specified dataset in 'config' is not implemented

    raise NotImplementedError


# Create training and testing datasets for the current fold

training_dataset = dataset(data=training_data,num_sensor=config["input_sensor"],
                                channel=config["channel"])
testing_dataset = dataset(data=testing_data,num_sensor=config["input_sensor"],
                                    channel=config["channel"])
# Create training and testing dataloaders using the datasets

training_dataloader = DataLoader(
    training_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])
testing_dataloader = DataLoader(
    testing_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])



# Initialize the MobileNetV1 model
model = M5(num_classes=training_dataset.num_classes)
# model = MobileNetV2(input_layer=training_dataset.channel,num_classes=training_dataset.num_classes,model_width=config["model_width"])

# Set the device to use (GPU if available, otherwise CPU)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)
# Calculating the flops and parameters for the model


macs, params = get_model_complexity_info(model, (1, config["input_sensor"]*config["input_time_window"]), as_strings=True,
                                        print_per_layer_stat=False, verbose=False)
macs = '{:<8}'.format( macs).replace(" ", "")
params = '{:<8}'.format( params).replace(" ", "")



# Define the optimizer, loss criterion, and scheduler

optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
criterion = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config["T_max"])



# Initialize variables to track the best accuracy achieved

best_acc = 0

# Define the training function

def train(epoches,model,optimizer,criterion,dataloader):
    """
    Trains a machine learning model using the provided data.

    Args:
        epochs (int): Number of training epochs.
        model: The machine learning model to be trained.
        optimizer: The optimizer used for updating model parameters.
        criterion: The loss function.
        dataloader: A data loader that provides batches of training data.

    Returns:
        Tuple: Average loss per batch, overall accuracy on the training data.
    """

    model.train()  # Sets the model in training mode
    correct = 0  # Counter for correct predictions
    total = 0  # Counter for total samples
    running_loss = 0.0  # Accumulator for loss during training
    with tqdm(total=len(dataloader)) as pbar:  # Progress bar for training iterations
        
        for i, data in enumerate(dataloader, 0):
            
            inputs, labels = data  # Extract inputs and labels from the current batch
            labels = labels.type(torch.LongTensor)
            # Prepare inputs and labels for computation on the appropriate device
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad() 
            # Forward pass: compute model outputs
            outputs = model(inputs)  
            # Determine predicted classes
            _, predicted = torch.max(outputs, 1)  
            # Calculate the loss
            loss = criterion(outputs, labels)  
            # Backward pass: compute gradients
            loss.backward()  
            # Update model parameters based on gradients
            optimizer.step()  
            # Increment the total number of samples
            total += labels.size(0)  
            # Increment correct predictions
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()  # Accumulate the loss
            pbar.update()
            pbar.set_description(f"Epoch: {epoches} | Loss: {running_loss/(i+1):.4f} | ACC: {100 * correct / total:.4f}%")
    print('Finished Training')
    # Calculate and return average loss per batch and overall accuracy
    return round(running_loss/len(dataloader),4), round(100 * correct / total,4)
    
#Define the testing function

def test(epoch,model,criterion,dataloader):
    """
    Performs testing (evaluation) of a machine learning model using the provided data.

    Args:
        epoch (int): Current epoch number.
        model: The machine learning model to be tested.
        criterion: The loss function.
        dataloader: A data loader that provides batches of testing data.

    Returns:
        Tuple: Average loss per batch, overall accuracy on the testing data.
    """
    # Sets the model in evaluation mode
    model.eval() 
    # Counter for correct predictions
    correct = 0
    # Counter for total samples
    total = 0
    # Accumulator for loss during testing
    running_loss = 0.0
    # Progress bar for testing iterations
    with tqdm(total=len(dataloader)) as pbar:
        # Disables gradient calculation during testing
        with torch.no_grad():
            for i, data in enumerate(dataloader, 0):
                # Extract inputs and labels from the current batch
                inputs, labels = data

                # Prepare inputs and labels for computation on the appropriate device
                labels = labels.type(torch.LongTensor)
                inputs, labels = inputs.to(device), labels.to(device)
                # Forward pass: compute model outputs
                outputs = model(inputs)
                # Determine predicted classes
                _, predicted = torch.max(outputs, 1)
                # Calculate the loss
                loss = criterion(outputs, labels)
                # Increment the total number of samples
                total += labels.size(0)
                # Increment correct predictions
                correct += (predicted == labels).sum().item()
                # Accumulate the loss
                running_loss += loss.item()
                # Update the progress bar
                pbar.update()
                pbar.set_description(f"Epoch: {epoch} | Loss: {running_loss/(i+1):.4f} | ACC: {100 * correct / total:.4f}%")
    
    print('Finished validation')
    # Calculate and return average loss per batch and overall accuracy
    return round(running_loss/len(dataloader),4), round(100 * correct / total,4)


epochs = config["Epoch"]

for epoch in range(epochs):
    
    train_loss,train_acc = train(epoch,model,optimizer,criterion,training_dataloader)
    test_loss,test_acc = test(epoch,model,criterion,testing_dataloader)
    

    if not os.path.isdir(config["model_save"]):
        os.makedirs(config["model_save"])
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(),f"{config['model_save']}/{type(model).__name__}_Param@{params}_MAC@{macs}_Acc@{best_acc:.3f}.pt")
        
    scheduler.step()
    epoch+=1
