import torch
import torch.nn.functional as F

# TODO g) Import your CenterSurroundConvolution layer for use here.
# from center_surround_convolution import CenterSurroundConvolution


# This linear fully connected network is just for reference. You can train it
# by calling:
#      python3 train_network --model fc
class FullyConnectedNetwork(torch.nn.Module):
    def __init__(self):
        super(FullyConnectedNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(28*28, 10)

    def forward(self, input_images):
        block = torch.flatten(input_images, 1)
        block = self.fc1(block)
        result = torch.nn.functional.log_softmax(block, dim=1)
        return result

# a) train this network by calling
#      python3 train_network --model conv
class TraditionalConvolutionalNetwork(torch.nn.Module):
    def __init__(self):
        super(TraditionalConvolutionalNetwork, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=16,
                                     kernel_size=3, stride=1)
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32,
                                     kernel_size=3, stride=1)
        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=64,
                                     kernel_size=3, stride=1)
        self.dropout1 = torch.nn.Dropout2d(0.25)
        self.fc1 = torch.nn.Linear(64, 10)

    def forward(self, input_images):
        block = self.conv1(input_images)
        block = torch.nn.functional.relu(block)
        block = torch.nn.functional.max_pool2d(block, 2)
        block = self.conv2(block)
        block = torch.nn.functional.relu(block)
        block = torch.nn.functional.max_pool2d(block, 2)
        block = self.conv3(block)
        block = torch.nn.functional.relu(block)
        block = torch.nn.functional.max_pool2d(block, 2)
        block = self.dropout1(block)
        block = torch.flatten(block, 1)
        block = self.fc1(block)
        result = torch.nn.functional.log_softmax(block, dim=1)
        return result

# g) In networks.py change the CenterSurroundConvolutionalNetwork to use your
# CenterSurround- Convolution layers instead of the Conv2d layers.
# Train the network by calling:
#   python3 train_network --model csc
class CenterSurroundConvolutionalNetwork(torch.nn.Module):
    def __init__(self):
        super(CenterSurroundConvolutionalNetwork, self).__init__()
        # TODO: Replace the Conv2d layers with your CenterSurroundConvolution
        # layers
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=16,
                                     kernel_size=3, stride=1)
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32,
                                     kernel_size=3, stride=1)
        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=64,
                                     kernel_size=3, stride=1)
        self.dropout1 = torch.nn.Dropout2d(0.25)
        self.fc1 = torch.nn.Linear(64, 10)

    def forward(self, input_images):
        block = self.conv1(input_images)
        block = torch.nn.functional.relu(block)
        block = torch.nn.functional.max_pool2d(block, 2)
        block = self.conv2(block)
        block = torch.nn.functional.relu(block)
        block = torch.nn.functional.max_pool2d(block, 2)
        block = self.conv3(block)
        block = torch.nn.functional.relu(block)
        block = torch.nn.functional.max_pool2d(block, 2)
        block = self.dropout1(block)
        block = torch.flatten(block, 1)
        block = self.fc1(block)
        result = torch.nn.functional.log_softmax(block, dim=1)
        return result
