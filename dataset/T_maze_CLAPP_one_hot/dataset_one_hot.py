# Import the PyTorch library, which is a popular deep learning framework.
# 'torch.nn' provides tools for building neural networks.
# 'Dataset' from 'torch.utils.data' is a base class for creating custom datasets in PyTorch.
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# Define a custom dataset class called 'Dataset_One_Hot'.
# This class is designed to load features and labels from files and provide them as a PyTorch Dataset.
# It is particularly useful when your data is stored in separate files for features and labels.
class Dataset_One_Hot(Dataset):
    # The '__init__' method is the constructor of the class.
    # It is called when an instance of the class is created.
    # Parameters:
    #   - features_file: Path to the file containing the features (input data).
    #   - labels_file: Path to the file containing the labels (target data).
    #   - device: The device (e.g., 'cpu' or 'cuda') where the tensors should be loaded.
    #   - transforms: Optional transformations to apply to the features.
    #   - target_transform: Optional transformations to apply to the labels.
    def __init__(self, features_file, labels_file, device, transforms=None, target_transform=None):
        # Store the file paths for features and labels.
        self.features_file = features_file
        self.labels_file = labels_file
        # Store the optional transformations.
        self.transforms = transforms
        self.target_transform = target_transform
        # Load the features and labels from the files and map them to the specified device.
        # 'map_location=device' ensures the tensors are loaded on the correct device (CPU/GPU).
        self.features = torch.load(self.features_file, map_location=device)
        self.labels = torch.load(self.labels_file, map_location=device)

    # The '__len__' method returns the number of samples in the dataset.
    # This is required by the PyTorch Dataset class.
    # It is used to know how many samples are available for training or evaluation.
    def __len__(self):
        return len(self.labels)

    # The '__getitem__' method retrieves a sample from the dataset at the given index.
    # This is required by the PyTorch Dataset class.
    # It is used to access individual samples during training or evaluation.
    # Parameters:
    #   - index: The index of the sample to retrieve.
    # Returns:
    #   - A tuple (features, label) corresponding to the sample at the given index.
    def __getitem__(self, index):
        # Retrieve the features and label at the specified index.
        features = self.features[index]
        label = self.labels[index]
        # Apply the optional transformations to the features and label, if provided.
        if self.transforms:
            features = self.transforms(features)
        if self.target_transform:
            label = self.target_transform(label)
        # Return the transformed features and label.
        return features, label