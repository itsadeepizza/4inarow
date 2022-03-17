import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class AIPlayer():

    def get_scores(self, batchboard):
        return None

    def play(self, batchboard):
        Q = self.get_scores(batchboard)
        move = Q.argmax(dim=1)
        return move


class NNPlayer(AIPlayer):
    def __init__(self, model: nn.Module):
        self.model = model

    def get_scores(self, batchboard):
        with torch.no_grad():
            Q = self.model.forward(batchboard.state)
        return Q



class ConvNet(nn.Module):
    def __init__(self, rows=6, cols=7):
        ch1 = 60
        super(ConvNet, self).__init__()
        self.l1 = nn.Conv2d(3, out_channels=ch1, kernel_size=5, groups=3, padding=2)
        self.l2 = nn.Conv2d(ch1, out_channels=120, kernel_size=3, padding=1)
        self.l3 = nn.Conv2d(120, 60, kernel_size=1)
        self.l4 = nn.Linear(60 * rows * cols, cols)


    def forward(self, x):
        x = torch.stack([x < 0, x == 0, x > 0], dim=1).float()
        x = self.l1(x)
        x = torch.relu(x)
        x = self.l2(x)
        x = torch.relu(x)
        x = self.l3(x)
        x = torch.relu(x)
        x = self.l4(x.flatten(1))
        return x


class smallDQN(nn.Module):

    def __init__(self, rows=6, cols=7):
        super(smallDQN, self).__init__()
        self.l1 = nn.Linear(rows * cols, 30)
        self.l2 = nn.Linear(30, 30)
        self.l3 = nn.Linear(30, 30)
        self.l4 = nn.Linear(30, cols)

        # Define proportion or neurons to dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, grid):
        # Input a grid m x n
        x = self.l1(grid.flatten(1))
        x = torch.sigmoid(x)
        x = self.l2(x)
        x = self.dropout(x)
        x = torch.sigmoid(x)
        x = self.l3(x)
        x = torch.sigmoid(x)
        x = self.l4(x)
        #x = torch.sigmoid(x)
        # Output a n array
        return x


class DQN(nn.Module):

    def __init__(self, rows=6, cols=7):
        super(DQN, self).__init__()
        self.l1 = nn.Linear(rows * cols, 300)
        self.l2 = nn.Linear(300, 500)
        self.l3 = nn.Linear(500, 50)
        self.l4 = nn.Linear(50, 25)
        self.l5 = nn.Linear(25, cols)

        # Define proportion or neurons to dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, grid):
        # Input a grid m x n
        x = self.l1(grid.flatten(1))
        x = torch.sigmoid(x)
        x = self.l2(x)
        x = self.dropout(x)
        x = torch.sigmoid(x)
        x = self.l3(x)
        x = self.dropout(x)
        x = torch.sigmoid(x)
        x = self.l4(x)
        x = torch.sigmoid(x)
        x = self.l5(x)
        #x = torch.sigmoid(x)
        # Output a n array
        return x


class channel_DQN(nn.Module):
    """Model with coins separated in 2 channels (one for each player)"""
    def __init__(self, rows=6, cols=7):
        ch1 = 50
        super(channel_DQN, self).__init__()
        self.l1 = nn.Conv2d(2, out_channels=ch1, kernel_size=4, padding=2)
        self.l2 = nn.Linear(7 * 8 * ch1, 300)
        self.l3 = nn.Linear(300, 300)
        self.l4 = nn.Linear(300, cols)


        # Define proportion or neurons to dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, grid):
        # Input a grid m x n
        double_channel = torch.zeros(grid.shape, device=grid.device).unsqueeze(1).repeat(1, 2, 1, 1)
        double_channel[:, 0, :, :][grid > 0] = 1
        double_channel[:, 1, :, :][grid < 0] = 1
        x = self.l1(double_channel)
        x = torch.sigmoid(x)
        x = self.dropout(x)
        x = self.l2(x.flatten(1))
        x = self.dropout(x)
        x = torch.sigmoid(x)
        x = self.l3(x)
        x = self.dropout(x)
        x = torch.sigmoid(x)
        x = self.l4(x) - 3
        #x = torch.sigmoid(x)
        # Output a n array
        return x





class full_channel_DQN(nn.Module):
    """Model with coins separated in 2 channels (one for each player)"""
    def __init__(self, rows=6, cols=7):
        ch1 = 50
        super(full_channel_DQN, self).__init__()
        self.l1 = nn.Linear(rows * cols * 2, 300)
        self.l2 = nn.Linear(300, 500)
        self.l3 = nn.Linear(500, 100)
        self.l4 = nn.Linear(100, cols)


        # Define proportion or neurons to dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, grid):
        # Input a grid m x n
        double_channel = torch.zeros(grid.shape, device=grid.device).unsqueeze(1).repeat(1, 2, 1, 1)
        double_channel[:, 0, :, :][grid > 0] = 1
        double_channel[:, 1, :, :][grid < 0] = 1
        x = self.l1(double_channel.flatten(1))
        x = torch.sigmoid(x)
        x = self.dropout(x)
        x = self.l2(x)
        x = self.dropout(x)
        x = torch.sigmoid(x)
        x = self.l3(x)
        x = self.dropout(x)
        x = torch.sigmoid(x)
        x = self.l4(x)
        x = torch.sigmoid(x) * 6 - 3
        # Output a n array
        return x


class Bottleneck(nn.Module):
    def __init__(self, in_chans, expansion, act=torch.relu):
        super().__init__()
        mid = in_chans * expansion
        self.l1 = nn.Linear(in_chans, mid)
        self.l2 = nn.Linear(mid, mid)
        self.l3 = nn.Linear(mid, in_chans)
        self.act = act
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        in_x = x
        x = self.l1(x)
        x = self.act(x)
        x = self.l2(x)
        x = self.act(x)
        x = self.l3(x)
        x = self.act(x)
        x = in_x + x
        return self.drop(x)


class full_channel_DQN_v2(nn.Module):
    """Model with coins separated in 2 channels (one for each player)"""
    def __init__(self, rows=6, cols=7):
        ch1 = 50
        super().__init__()

        chans = 150

        self.l1 = nn.Linear(rows * cols, chans)
        self.b1 = Bottleneck(chans, 2)
        self.b2 = Bottleneck(chans, 2)
        self.l2 = nn.Linear(chans, cols)
        # Define proportion or neurons to dropout

    def forward(self, x):
        # Input a grid m x n
        x = self.l1(x.flatten(1))
        x = torch.sigmoid(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.l2(x)
        x = torch.sigmoid(x) * 6 - 3
        # Output a n array
        return x


class conv_DQN(nn.Module):

    def __init__(self, rows=6, cols=7):
        ch1 = 50
        super(conv_DQN, self).__init__()
        self.l1 = nn.Conv2d(1, out_channels=ch1, kernel_size=4, padding=2)
        self.l2 = nn.Linear(7 * 8 * ch1, 100)
        self.l3 = nn.Linear(100, cols)


        # Define proportion or neurons to dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, grid):
        # Input a grid m x n
        x = self.l1(grid.unsqueeze(1))
        x = torch.sigmoid(x)
        x = self.dropout(x)
        x = self.l2(x.flatten(1))
        #x = self.dropout(x)
        x = torch.sigmoid(x)
        x = self.l3(x)
        #x = torch.sigmoid(x)
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
