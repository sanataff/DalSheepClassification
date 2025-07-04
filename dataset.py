import os
import pandas as pd
from torchvision.io import decode_image

from torch.utils.data import Dataset, DataLoader
import PIL
from PIL import Image

class SheepDataset(Dataset):
    
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = annotations_file
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = decode_image(img_path)
        image = Image.open(img_path)
        
        
  
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
   

    def __len__(self):
        return len(self.img_labels)