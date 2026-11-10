import torch
import torch.nn as nn


class ASLLandmarkNet(nn.Module):

    def __init__(self, num_classes=28):

        super().__init__()

        self.network = nn.Sequential(

            # Input: 63 landmark features
            nn.Linear(63, 128),

            nn.ReLU(),

            nn.BatchNorm1d(128),

            nn.Dropout(0.30),


            # Hidden layer
            nn.Linear(128, 64),

            nn.ReLU(),

            nn.BatchNorm1d(64),

            nn.Dropout(0.25),


            # Output
            nn.Linear(64, num_classes)
        )


    def forward(self, x):

        return self.network(x)