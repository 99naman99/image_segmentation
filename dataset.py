import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir) #returns a list of all files and directories inside the specified image_dir
        self.transform = transform

        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.gif"))
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        #mask[mask == 255] = 1.0

        return self.transform(image), self.transform(mask) 
        
           

         
            
        