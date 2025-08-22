import torch
import torch.nn as nn

class PCA(nn.Module):
    """
    Principal Component Analysis (PCA) implemented as a PyTorch Module.
    This implementation is inspired by:
    https://github.com/gngdb/pytorch-pca/blob/main/pca.py

    PCA is a dimensionality reduction technique that transforms data to a new coordinate system
    such that the greatest variance lies on the first axis (first principal component),
    the second greatest variance on the second axis, and so on.

    Attributes:
        size_input (int): Dimensionality of the input data
        num_components (int): Number of principal components to keep
        mean (torch.Tensor): Mean of the training data (computed during fit)
        Vh (torch.Tensor): Principal components (eigenvectors) of the data
    """

    def __init__(self, size_input, num_components, *args, **kwargs):
        """
        Initialize the PCA module.

        Args:
            size_input (int): Dimensionality of the input data
            num_components (int): Number of principal components to keep
                                  (must be <= min(size_input, number of samples))
            *args: Additional arguments for nn.Module
            **kwargs: Additional keyword arguments for nn.Module
        """
        super().__init__(*args, **kwargs)

        # --- Store hyperparameters ---
        self.size_input = size_input  # Original dimensionality of the input data
        self.num_components = num_components  # Number of principal components to keep

        # --- Attributes to be set during fitting ---
        # These will store the learned parameters after calling fit()
        self.mean = None  # Will store the mean of the training data
        self.Vh = None  # Will store the principal components (transposed)

    def fit(self, X):
        """
        Fit the PCA model to the training data.

        Args:
            X (torch.Tensor): Training data of shape (n_samples, n_features)
        """
        n, d = X.shape  # n = number of samples, d = number of features

        # --- Validate input dimensions ---
        # Ensure we don't ask for more components than we have features or samples
        assert self.num_components <= d and self.num_components <= n, \
            'num_components must be <= min(n_features, n_samples)'

        # --- Step 1: Center the data by subtracting the mean ---
        # Compute mean along features (dim=0) and keep dimensions for broadcasting
        self.mean = X.mean(dim=0, keepdims=True)
        Z = X - self.mean  # Centered data

        # --- Step 2: Compute Singular Value Decomposition (SVD) ---
        # We use SVD instead of eigendecomposition for better numerical stability
        # full_matrices=False returns U, S, Vh with reduced dimensions
        U, S, Vh = torch.linalg.svd(Z, full_matrices=False)

        # --- Step 3: Store the principal components ---
        # Vh contains the principal components in its rows
        # We only keep the first num_components rows (components with highest variance)
        self.Vh = Vh[:self.num_components, :]

    def forward(self, data):
        """
        Transform the input data to the principal component space.

        Args:
            data (torch.Tensor): Input data of shape (n_samples, n_features)

        Returns:
            torch.Tensor: Transformed data of shape (n_samples, num_components)

        Raises:
            AssertionError: If fit() hasn't been called yet
        """
        # --- Check if the model has been fitted ---
        assert hasattr(self, 'Vh'), 'Need to call fit() before transforming data'

        # --- Step 1: Center the data using the stored mean ---
        centered_data = data - self.mean

        # --- Step 2: Project the centered data onto the principal components ---
        # This is equivalent to matrix multiplication: (data - mean) @ Vh.T
        # Each row of Vh is a principal component, so we transpose for the dot product
        return centered_data @ self.Vh.T

    def transform(self, data):
        """
        Alias for forward() to match scikit-learn's PCA interface.
        """
        return self.forward(data)

    def inverse_transform(self, data):
        """
        Transform data back to its original space.

        Args:
            data (torch.Tensor): Data in principal component space of shape (n_samples, num_components)

        Returns:
            torch.Tensor: Data in original space of shape (n_samples, size_input)
        """
        assert hasattr(self, 'Vh'), 'Need to call fit() before inverse transforming data'
        # Reconstruct the original data (approximation) from the reduced representation
        return (data @ self.Vh) + self.mean