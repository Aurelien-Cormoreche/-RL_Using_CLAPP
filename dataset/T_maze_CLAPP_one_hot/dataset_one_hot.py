import torch
import torch.nn as nn
from torch.utils.data import Dataset

class Dataset_One_Hot(Dataset):
    def __init__(self, features_file, labels_file, device, transforms = None, target_transform = None):
        self.features_file = features_file
        self.labels_file = labels_file
        self.transforms = transforms
        self.target_transform = target_transform
        self.features = torch.load(self.features_file, map_location= device)
        self.labels = torch.load(self.labels_file, map_location= device)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        features = self.features[index]
        label = self.labels[index]
        if self.transforms:
            features = self.transforms(features)
        if self.target_transform:
            label = self.target_transform(label)
        return features, label
    

        


       