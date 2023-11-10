import os
from glob import glob
from pathlib import Path
import shutil
import numpy as np
import csv
import torch
import torch.utils.data
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import torchvision.datasets as datasets
from torchvision.datasets import CIFAR10
from phenobench_anomaly.datasets.phenobench_anomaly_dataset import PhenoBenchAnomalyDataset
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


class PhenoBenchAnomalyDataset_DDAD(PhenoBenchAnomalyDataset):
    def __init__(self, root_dir, split, weed_percentage, config, transform=None, overfit=False):
        super().__init__(root_dir, split, weed_percentage, transform=None, overfit=overfit)
        self.transform = A.Compose([
            A.Resize(config.data.image_size, config.data.image_size),
            A.Normalize(mean=0.5, std=0.5),
            # ToTensorV2()
        ])

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        label = 'good'
        if self.split == "train":
            return sample["image"], label
        elif self.split == "val":
            ano_mask = np.zeros_like(sample["semantics"])
            ano_mask[sample["semantics"] == 2] = 1
            if np.sum(ano_mask) > 0:
                label = 'defective'
            ano_mask = np.expand_dims(ano_mask, axis=0)
            return sample["image"], ano_mask, label


class Dataset_maker(torch.utils.data.Dataset):
    def __init__(self, root, category, config, is_train=True):
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((config.data.image_size, config.data.image_size)),  
                transforms.ToTensor(), # Scales data into [0,1] 
                transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1] 
            ]
        )
        self.config = config
        self.mask_transform = transforms.Compose(
            [
                transforms.Resize((config.data.image_size, config.data.image_size)),
                transforms.ToTensor(), # Scales data into [0,1] 
            ]
        )
        if is_train:
            if category:
                self.image_files = glob(
                    os.path.join(root, category, "train", "good", "*.png")
                )
            else:
                self.image_files = glob(
                    os.path.join(root, "train", "good", "*.png")
                )
        else:
            if category:
                self.image_files = glob(os.path.join(root, category, "test", "*", "*.png"))
            else:
                self.image_files = glob(os.path.join(root, "test", "*", "*.png"))
        self.is_train = is_train

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = self.image_transform(image)
        if(image.shape[0] == 1):
            image = image.expand(3, self.config.data.image_size, self.config.data.image_size)
        if self.is_train:
            label = 'good'
            return image, label
        else:
            if self.config.data.mask:
                if os.path.dirname(image_file).endswith("good"):
                    target = torch.zeros([1, image.shape[-2], image.shape[-1]])
                    label = 'good'
                else :
                    if self.config.data.name == 'MVTec':
                        target = Image.open(
                            image_file.replace("/test/", "/ground_truth/").replace(
                                ".png", "_mask.png"
                            )
                        )
                    else:
                        target = Image.open(
                            image_file.replace("/test/", "/ground_truth/"))
                    target = self.mask_transform(target)
                    label = 'defective'
            else:
                if os.path.dirname(image_file).endswith("good"):
                    target = torch.zeros([1, image.shape[-2], image.shape[-1]])
                    label = 'good'
                else :
                    target = torch.zeros([1, image.shape[-2], image.shape[-1]])
                    label = 'defective'
                
            return image, target, label

    def __len__(self):
        return len(self.image_files)
