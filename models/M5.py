import torch
import torch.nn as nn
import torch.nn.functional as F

class M5(nn.Module):
    def __init__(self, num_classes=7):
        super(M5, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=16, 
            kernel_size=(16, 1), 
            stride=(16, 1)
        )
        # First max pooling layer
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(
            in_channels=16, 
            out_channels=16, 
            kernel_size=(3, 1), 
            stride=(1, 1)
        )
        # Second max pooling layer
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        
        # Third convolutional layer
        self.conv3 = nn.Conv2d(
            in_channels=16, 
            out_channels=32, 
            kernel_size=(3, 1), 
            stride=(1, 1)
        )
        
        # Adaptive average pooling to reduce the feature map to size (1, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layer
        self.fc = nn.Linear(32, num_classes)
        
    def forward(self, x):
        # Reshape input to (batch_size, 1, 3840, 1)
        x = x.view(-1, 1, 3840, 1)
        
        # First convolutional block
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        # Second convolutional block
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        # Third convolutional block
        x = F.relu(self.conv3(x))
        
        # Adaptive average pooling
        x = self.avgpool(x)
        
        # Flatten the tensor
        x = x.view(x.size(0), -1)
        
        # Fully connected layer
        x = self.fc(x)
        
        # Output layer (Identity)
        return x

# # Initialize model and test
# test = M5()
# with torch.no_grad():
#     data = torch.randn(1, 3840)  # Sample input tensor
#     prediction = test(data)
#     print(prediction)  # Output predictions
