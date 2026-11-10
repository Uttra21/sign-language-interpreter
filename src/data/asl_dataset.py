import pandas as pd
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ASLDataset(Dataset):

    def __init__(self, csv_file, image_size=64):

        # Load CSV
        self.data = pd.read_csv(csv_file)

        # Image transformation
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        # Get image path
        image_path = self.data.iloc[index]["image_path"]

        # Get label
        label = int(self.data.iloc[index]["label_id"])

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Apply transformations
        image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)