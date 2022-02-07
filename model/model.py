import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class DQN(nn.Module):

    def __init__(self, rows=6, cols=7):
        super(DQN, self).__init__()
        self.l1 = nn.Linear(rows * cols, 50)
        self.l2 = nn.Linear(50, 50)
        self.l3 = nn.Linear(50, 25)
        self.l4 = nn.Linear(25, cols)

    def forward(self, grid):
        # Input a grid m x n
        x = self.l1(grid.flatten(1))
        x = torch.sigmoid(x)
        x = self.l2(x)
        x = torch.sigmoid(x)
        x = self.l3(x)
        x = torch.sigmoid(x)
        x = self.l4(x)
        x = torch.sigmoid(x)
        # Output a n array
        return x

"""
if __name__ == "__main__":
    model = DQN
    testgrid = []
    for row in range(6):
        el
        for col in range(7):
"""


"""
class DQN(nn.Module):

    def __init__(self, rows=6, cols=7):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, cols)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        # Input a grid m x n
        # Output a n array

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))
"""
