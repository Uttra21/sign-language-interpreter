import torch
import torch.nn as nn


class ASLCNN(nn.Module):

    def __init__(self, num_classes=29):

        super().__init__()

        # -----------------------------
        # Convolutional feature extractor
        # -----------------------------

        self.features = nn.Sequential(

            # 64 x 64 -> 32 x 32
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 32 x 32 -> 16 x 16
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 16 x 16 -> 8 x 8
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Dropout(0.25)
        )

        # -----------------------------
        # Fully connected classifier
        # -----------------------------

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                128 * 8 * 8,
                256
            ),

            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(
                256,
                num_classes
            )
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x