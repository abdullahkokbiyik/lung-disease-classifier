import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv3 = nn.Conv2d(20, 20, kernel_size=5, padding=3)
        self.conv4 = nn.Conv2d(20, 20, kernel_size=3)
        self.mp = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(2880, 1440)
        self.fc2 = nn.Linear(1440, 720)
        self.fc3 = nn.Linear(720, 81)
        self.fc4 = nn.Linear(81, 3)
    
    
    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = F.relu(self.mp(self.conv2(x)))
        x = F.relu(self.mp(self.conv3(x)))
        x = F.relu(self.mp(self.conv4(x)))
        # x = self.dropout(x)
        x = x.view(in_size, -1) 
        x = self.dropout(self.fc1(x))
        x = self.dropout(self.fc2(x))
        x = self.dropout(self.fc3(x))
        x = self.dropout(self.fc4(x))
        return F.log_softmax(x)
        