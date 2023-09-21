

from torch.utils.data import DataLoader
from tqdm import tqdm
from data_preprocessing import ICE_Lab_dataset
# from models.mobilenetv2 import MobileNetV2
from models.mobilenetv1 import MobilenetV1
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
    data,label,num_classes = utils().extra_data(config["data_path"])

    # Assign the data path from 'config' to 'root' for further usage
    root = (data, label, num_classes)
else:

    # Raise an exception if the specified dataset in 'config' is not implemented

    raise NotImplementedError

# Initialize lists for training and testing dataloaders

training_dataloaders = []
testing_dataloaders = []

# Iterate over each fold


# Create training and testing datasets for the current fold

training_dataset = dataset(root=root,width=config["input_width"],
                                height=config["input_height"],channel=config["channel"],train=True)
testing_dataset = dataset(root=root,width=config["input_width"],
                                    height=config["input_height"], channel=config["channel"], train=False)
# Create training and testing dataloaders using the datasets

training_dataloader = DataLoader(
    training_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])
testing_dataloader = DataLoader(
    testing_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])



# Initialize the MobileNetV1 model
model = MobilenetV1(ch_in=training_dataset.channel,
                    n_classes=training_dataset.num_classes,
                    Global_ratio=config["model_width"])
# model = MobileNetV2(input_layer=training_dataset.channel,num_classes=training_dataset.num_classes,model_width=config["model_width"])

# Set the device to use (GPU if available, otherwise CPU)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)
# Calculating the flops and parameters for the model


macs, params = get_model_complexity_info(model, (1, 8, 24), as_strings=True,
                                        print_per_layer_stat=False, verbose=False)
macs = '{:<8}'.format( macs)
params = '{:<8}'.format( params)



# Define the optimizer, loss criterion, and scheduler

optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],weight_decay=config["weight_decay"])
criterion = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config["T_max"])
model_orginal_weight = model.state_dict().copy()


if config["finetune"]:
    model.load_state_dict(torch.load(config["pretrain_model_path"]),strict=False)
    # frezze the first couple layer for funeting
    for conv1_param in model.conv1.parameters():
        conv1_param.requires_grad = False
    for bn1_param in model.bn1.parameters():
        bn1_param.requires_grad = False
    for block0_param in model.layers[0].parameters():
        block0_param.requires_grad = False
    model_orginal_weight = model.state_dict().copy()

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

#Iterate over the training and testing dataloaders for each fold


    # refresh the model weight
model.load_state_dict(model_orginal_weight)
for epoch in range(epochs):
    
    train_loss,train_acc = train(epoch,model,optimizer,criterion,training_dataloader)
    test_loss,test_acc = test(epoch,model,criterion,testing_dataloader)
    

    if not os.path.isdir(config["model_save"]):
        os.makedirs(config["model_save"])
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(),f"{config['model_save']}/MobilenetV1_Param@{params}_MAC@{macs}_Acc@{best_acc:.3f}.pt")
        
    scheduler.step()
    epoch+=1

#Calculate the final average training and testing accuracies
print(
    f"Final testing Acc:{sum(test_acces)/len(test_acces)} | Final training Acc:{sum(train_acces)/len(train_acces)}")
