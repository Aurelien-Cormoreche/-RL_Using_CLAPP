import seaborn as sns
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from utils.load_standalone_model import load_model
import os
import gymnasium as gym
from torchvision.models import resnet50, ResNet50_Weights

class TmazeDiscretizer:
    """
    Class for discretizing and analyzing a T-maze environment.
    Extracts features from different positions and orientations in the maze,
    computes similarity matrices, and visualizes the results.

    Attributes:
        env: The T-maze environment
        encoder: Neural network encoder for feature extraction
        featureslist: List storing extracted features
        discrete_positions: Predefined discrete positions in the maze
        resize: Flag indicating if input needs resizing for the encoder
    """

    def __init__(self, env, encoder=None, encoder_type='CLAPP'):
        """
        Initialize the TmazeDiscretizer.

        Args:
            env: The T-maze environment
            encoder: Neural network encoder for feature extraction
            encoder_type: Type of encoder ('CLAPP' or other)
        """
        self.env = self._unwrap_env(env)
        self.device = "cuda" if torch.cuda.is_available() else "mps"
        self.encoder = encoder.to(self.device) if encoder is not None else None
        self.featureslist = []
        self.resize = encoder_type != 'CLAPP'  # Resize needed for non-CLAPP encoders

        # Generate discrete positions in the T-maze
        self.discrete_positions = self._generate_discrete_positions()

    def _generate_discrete_positions(self):
        """
        Generate discrete positions in the T-maze.
        These positions represent key locations in the environment.

        Returns:
            List of discrete positions in the maze
        """
        positions = []

        # Main corridor (room1) - bottom and top walls
        for x in range(3):
            positions.append([1.37*(2*x+1), 0, 0])  # Bottom wall
            positions.append([1.37*(2*x+1), 0, 0])  # Top wall (duplicate?)

        # Left and right arms (room2) - back wall
        for x in range(5):
            positions.append([9.78, 0, 1.37*(2*x+1)-6.85])

        # Junction between room1 and room2
        for x in [0, 1, 3, 4]:  # Exclude central position (x=2)
            positions.append([8.5, 0, 1.37*(2*x+1)-6.85])

        # Corners of the arms
        positions.append([9.78, 0, -6.0])  # Left arm corner
        positions.append([9.78, 0, 6.0])   # Right arm corner

        print(positions)
        return positions

    def get_grid_positions(self, resolution=0.5):
        """
        Generate a grid of positions in the navigable space.

        Args:
            resolution: Distance between grid points

        Returns:
            List of grid positions in the maze
        """
        grid_positions = []

        # Room1: main corridor
        x_range = np.arange(-0.22, 8, resolution)
        z_range = np.arange(-1.37, 1.37, resolution)

        for x in x_range:
            for z in z_range:
                grid_positions.append([x, 1.37, z])

        # Room2: T arms
        x_range = np.arange(8, 10.74, resolution)
        z_min = -6.85 if self.env.left_arm else -1.37
        z_max = 6.85 if self.env.right_arm else 1.37
        z_range = np.arange(z_min, z_max, resolution)

        for x in x_range:
            for z in z_range:
                grid_positions.append([x, 1.37, z])

        return grid_positions

    def extract_features(self, obs):
        """
        Extract features from an observation using the encoder.

        Args:
            obs: Observation/image from the environment

        Returns:
            Extracted features as numpy array
        """
        if self.encoder is not None:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
            if self.resize:
                # Reshape for non-CLAPP models (e.g., ResNet)
                obs_tensor = obs_tensor.view(1, obs_tensor.shape[2], obs_tensor.shape[0], obs_tensor.shape[1])

            with torch.no_grad():
                features = self.encoder(obs_tensor)

            return features.cpu().numpy()

    def extract_features_from_all_positions(self, positions=None, orientations=None):
        """
        Extract features from all positions and orientations in the maze.

        Args:
            positions: List of positions to test (default: self.discrete_positions)
            orientations: List of orientations in degrees (default: 8 orientations)

        Returns:
            Array of extracted features
        """
        if positions is None:
            positions = self.discrete_positions

        if orientations is None:
            orientations = [0, 45, 90, 135, 180, 225, 270, 315]  # 8 orientations

        self.featureslist = []
        position_orientation_pairs = []
        wsh = self.env.reset()
        i = 0

        for pos_idx, pos in enumerate(positions):
            for orient in orientations:
                try:
                    # Get observation at current position and orientation
                    wsh = self.env.render_obs()

                    # Convert to grayscale if needed
                    wsh = np.sum(
                        np.multiply(wsh, np.array([0.2125, 0.7154, 0.0721])), axis=-1, keepdims=True
                    ).astype(np.uint8)

                    # Extract features
                    features = self.extract_features(wsh)

                    # Set agent position and orientation
                    self.env.agent.pos = pos
                    self.env.agent.dir = np.deg2rad(orient)

                    # Store features
                    features = features.flatten()
                    self.featureslist.append(features)
                    position_orientation_pairs.append((pos_idx, orient))

                    i += 1
                except Exception as e:
                    print(f"Error at position {pos} with orientation {orient}: {e}")
                    continue

        self.featureslist = np.array(self.featureslist)
        self.position_orientation_pairs = position_orientation_pairs

        return self.featureslist

    def compute_similarity_matrix(self, features=None):
        """
        Compute cosine similarity matrix between all features.

        Args:
            features: Features to use (default: self.featureslist)

        Returns:
            Cosine similarity matrix
        """
        if features is None:
            if len(self.featureslist) == 0:
                raise ValueError("No features extracted. Call extract_features_from_all_positions() first")
            features = self.featureslist

        # Compute cosine similarity
        similarity_matrix = cosine_similarity(features)
        return similarity_matrix

    def visualize_similarity_matrix(self, similarity_matrix=None, positions_list=None,
                                  show_orientations=False, save_path=None):
        """
        Visualize the similarity matrix as a heatmap.

        Args:
            similarity_matrix: Similarity matrix (computed if None)
            positions_list: List of positions (default: self.discrete_positions)
            show_orientations: If True, show orientations in labels
            save_path: Path to save the image (optional)
        """
        if similarity_matrix is None:
            similarity_matrix = self.compute_similarity_matrix()

        if positions_list is None:
            positions_list = self.discrete_positions

        plt.figure(figsize=(15, 12))

        # Create labels
        if show_orientations and hasattr(self, 'position_orientation_pairs'):
            labels = []
            for pos_idx, orient in self.position_orientation_pairs:
                pos = positions_list[pos_idx]
                labels.append(f"P{pos_idx}_{orient}°\n({pos[0]:.1f},{pos[2]:.1f})")
        else:
            labels = [f"P{i}\n({pos[0]:.1f},{pos[2]:.1f})" for i, pos in enumerate(positions_list)]

        # Adjust labels if too many
        if len(labels) != similarity_matrix.shape[0]:
            labels = [f"F{i}" for i in range(similarity_matrix.shape[0])]

        # Create heatmap
        sns.heatmap(similarity_matrix,
                    annot=False,
                    cmap='viridis',
                    xticklabels=labels if len(labels) <= 50 else False,
                    yticklabels=labels if len(labels) <= 50 else False,
                    cbar_kws={'label': 'Cosine Similarity'})

        plt.title('Cosine Similarity Matrix - Discretized T-maze')
        plt.xlabel('Positions/Orientations')
        plt.ylabel('Positions/Orientations')

        if len(labels) <= 50:
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def get_position_similarity_summary(self, positions_list=None):
        """
        Summarize similarity between positions (averaged over orientations).

        Args:
            positions_list: List of positions (default: self.discrete_positions)

        Returns:
            Similarity matrix between positions (averaged over orientations)
        """
        if positions_list is None:
            positions_list = self.discrete_positions

        if not hasattr(self, 'position_orientation_pairs'):
            raise ValueError("No orientation data available")

        n_positions = len(positions_list)
        position_similarity = np.zeros((n_positions, n_positions))

        # Group by position
        for i in range(n_positions):
            for j in range(n_positions):
                similarities = []
                for idx1, (pos1, orient1) in enumerate(self.position_orientation_pairs):
                    for idx2, (pos2, orient2) in enumerate(self.position_orientation_pairs):
                        if pos1 == i and pos2 == j:
                            sim_matrix = self.compute_similarity_matrix()
                            similarities.append(sim_matrix[idx1, idx2])

                if similarities:
                    position_similarity[i, j] = np.mean(similarities)

        return position_similarity

    def _unwrap_env(self, env):
        """
        Unwrap the environment to access the base environment.

        Args:
            env: The environment to unwrap

        Returns:
            The base environment
        """
        base_env = env
        while hasattr(base_env, 'env'):
            base_env = base_env.env
        return base_env

    def render_suspicious_positions(self, suspicious_indices=None):
        """
        Render suspicious positions (pairs with high similarity differences).

        Args:
            suspicious_indices: List of index pairs to render
        """
        if suspicious_indices is None:
            print("No suspicious indices provided.")
            return

        if not hasattr(self, 'position_orientation_pairs'):
            print("No position_orientation_pairs found.")
            return

        for idx_pair in suspicious_indices:
            i, j = idx_pair
            # Render first suspicious position
            pos_idx1, orient1 = self.position_orientation_pairs[i]
            pos1 = self.discrete_positions[pos_idx1]
            self.env.agent.pos = pos1
            self.env.agent.dir = orient1
            print(f"Rendering suspicious position {pos_idx1} with orientation {orient1} (index {i})")
            self.env.render()

            # Render second suspicious position
            pos_idx2, orient2 = self.position_orientation_pairs[j]
            pos2 = self.discrete_positions[pos_idx2]
            self.env.agent.pos = pos2
            self.env.agent.dir = orient2
            print(f"Rendering suspicious position {pos_idx2} with orientation {orient2} (index {j})")
            self.env.render()

def difference_matrix(matrix1, matrix2, threshold=1):
    """
    Compute difference matrix and find indices where difference exceeds threshold.

    Args:
        matrix1: First similarity matrix
        matrix2: Second similarity matrix
        threshold: Threshold for significant differences

    Returns:
        Difference matrix and list of indices where difference > threshold
    """
    matrix = np.abs(matrix1 - matrix2)
    below_threshold_indices = []

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] > threshold:
                below_threshold_indices.append((i, j))

    print("Indices with difference above threshold:", below_threshold_indices)
    return matrix, below_threshold_indices

if __name__ == '__main__':
    # Load models
    model_path = os.path.abspath('trained_models')
    encoder1 = load_model(model_path=model_path)
    encoder2 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    feature_dim = 1000

    # Register and create environment
    gym.envs.register(
        id='MyTMaze-v0',
        entry_point='envs.T_maze.custom_T_Maze_V0:MyTmaze'
    )

    envs = gym.make("MyTMaze", render_mode='human', visible_reward=False, reward=False, remove_images=True)

    # Create discretizers for both encoders
    TmazeforMatrix1 = TmazeDiscretizer(env=envs, encoder=encoder1)
    # TmazeforMatrix2 = TmazeDiscretizer(env=envs, encoder=encoder2, encoder_type='resnet')

    # Extract features and compute similarity for first encoder
    features1 = TmazeforMatrix1.extract_features_from_all_positions()
    matrice1 = TmazeforMatrix1.compute_similarity_matrix(features=features1)

    '''
    # Extract features and compute similarity for second encoder
    features2 = TmazeforMatrix2.extract_features_from_all_positions()
    matrice2 = TmazeforMatrix2.compute_similarity_matrix(features=features2)
    differencematrix, suspicious_indices = difference_matrix(matrice1, matrice2, threshold=1)
    '''

    # Visualize similarity matrix for first encoder
    TmazeforMatrix1.visualize_similarity_matrix(similarity_matrix=matrice1)
    # TmazeforMatrix1.render_suspicious_positions(suspicious_indices=suspicious_indices)