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
    def __init__(self, env, encoder=None, encoder_type='CLAPP'):
   
        self.env =  self._unwrap_env(env)
        self.device = "cuda" if torch.cuda.is_available() else "mps"
        self.encoder = encoder.to(self.device) if encoder is not None else None
        self.featureslist = []
        self.resize = encoder_type != 'CLAPP'
        
        # Positions discrétisées basées sur ton code
        self.discrete_positions = self._generate_discrete_positions()
        
    def _generate_discrete_positions(self):
        """Génère les positions discrétisées du T-maze"""
        positions = []
        
        # Corridor principal (room1) - murs du haut et du bas
        for x in range(3):
            positions.append([1.37*(2*x+1),0, 0])  # Mur du bas
            positions.append([1.37*(2*x+1), 0,0])   # Mur du haut
            
        # Bras gauche et droit (room2) - mur du fond
        for x in range(5):
            positions.append([9.78,0, 1.37*(2*x+1)-6.85])
            
        # Jonction entre room1 et room2
        for x in [0, 1, 3, 4]:  # Exclut la position centrale (x=2)
            positions.append([8.5, 0, 1.37*(2*x+1)-6.85])
            
        # Coins des bras
        positions.append([9.78, 0, -6.0])  # Coin bras gauche
        positions.append([9.78, 0, 6.0])   # Coin bras droit
        print(positions)
        return positions
    
    def get_grid_positions(self, resolution=0.5):
        """Génère une grille de positions dans l'espace navigable"""
        grid_positions = []
        
        # Room1: corridor principal
        x_range = np.arange(-0.22, 8, resolution)
        z_range = np.arange(-1.37, 1.37, resolution)
        
        for x in x_range:
            for z in z_range:
                grid_positions.append([x, 1.37, z])
        
        # Room2: bras du T
        x_range = np.arange(8, 10.74, resolution)
        z_min = -6.85 if self.env.left_arm else -1.37
        z_max = 6.85 if self.env.right_arm else 1.37
        z_range = np.arange(z_min, z_max, resolution)
        
        for x in x_range:
            for z in z_range:
                grid_positions.append([x, 1.37, z])
    
        return grid_positions
    
    def extract_features(self, obs):
        """Extrait les caractéristiques de l'observation"""
        if self.encoder is not None:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device) 
            if self.resize:
                obs_tensor = obs_tensor.view(1, obs_tensor.shape[2], obs_tensor.shape[0], obs_tensor.shape[1])  # Reshape pour le modèle
            with torch.no_grad():
                features = self.encoder(obs_tensor)
            return features.cpu().numpy()
    
    def extract_features_from_all_positions(self, positions=None, orientations=None):
        """
        Extrait les features de toutes les positions et orientations
        
        Args:
            positions: Liste des positions à tester (par défaut self.discrete_positions)
            orientations: Liste des orientations en degrés (par défaut [0, 45, 90, 135, 180, 225, 270, 315])
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
                # try:
                    # Reset et placer l'agent
                
                   
                    wsh = self.env.render_obs()
                    wsh = np.sum(
                        np.multiply(wsh, np.array([0.2125, 0.7154, 0.0721])), axis=-1
                    , keepdims= True).astype(np.uint8)
                    
                    features = self.extract_features(wsh)
                    self.env.agent.pos=pos
                    self.env.agent.dir = orient 
                    
                    features = features.flatten()
                    
                    self.featureslist.append(features)
                    position_orientation_pairs.append((pos_idx, orient))
                    
                    i += 1
                # except Exception as e:
                #     print(f"Erreur à la position {pos} avec orientation {orient}: {e}")
                #     continue
        
        self.featureslist = np.array(self.featureslist)
        self.position_orientation_pairs = position_orientation_pairs
        
        return self.featureslist
    
    def compute_similarity_matrix(self, features=None):
        """
        Calcule la matrice de similarité cosinus entre toutes les features
        
        Args:
            features: Features à utiliser (par défaut self.featureslist)
        
        Returns:
            Matrice de similarité cosinus
        """
        if features is None:
            if len(self.featureslist) == 0:
                raise ValueError("Aucune feature extraite. Appelez d'abord extract_features_from_all_positions()")
            features = self.featureslist
        
        # Calculer la similarité cosinus
        similarity_matrix = cosine_similarity(features)
        
        return similarity_matrix
    
    def visualize_similarity_matrix(self, similarity_matrix=None, positions_list=None, 
                                  show_orientations=False, save_path=None):
        """
        Visualise la matrice de similarité
        
        Args:
            similarity_matrix: Matrice de similarité (calculée automatiquement si None)
            positions_list: Liste des positions (par défaut self.discrete_positions)
            show_orientations: Si True, affiche les orientations dans les labels
            save_path: Chemin pour sauvegarder l'image (optionnel)
        """
        if similarity_matrix is None:
            similarity_matrix = self.compute_similarity_matrix()
        
        if positions_list is None:
            positions_list = self.discrete_positions
        
        plt.figure(figsize=(15, 12))
        
        # Créer des labels
        if show_orientations and hasattr(self, 'position_orientation_pairs'):
            labels = []
            for pos_idx, orient in self.position_orientation_pairs:
                pos = positions_list[pos_idx]
                labels.append(f"P{pos_idx}_{orient}°\n({pos[0]:.1f},{pos[2]:.1f})")
        else:
            labels = [f"P{i}\n({pos[0]:.1f},{pos[2]:.1f})" for i, pos in enumerate(positions_list)]
        
        # Adapter les labels à la taille de la matrice
        if len(labels) != similarity_matrix.shape[0]:
            labels = [f"F{i}" for i in range(similarity_matrix.shape[0])]
        
        sns.heatmap(similarity_matrix, 
                    annot=False,  # Désactiver les annotations pour plus de clarté
                    cmap='viridis',
                    xticklabels=labels if len(labels) <= 50 else False,  # Cacher les labels si trop nombreux
                    yticklabels=labels if len(labels) <= 50 else False,
                    cbar_kws={'label': 'Similarité cosinus'})
        
        plt.title('Matrice de Similarité cosinus - T-maze Discrétisé')
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
        Résume la similarité entre positions (moyenne sur les orientations)
        
        Returns:
            Matrice de similarité entre positions (moyennée sur les orientations)
        """
        if positions_list is None:
            positions_list = self.discrete_positions
        
        if not hasattr(self, 'position_orientation_pairs'):
            raise ValueError("Pas de données d'orientation disponibles")
        
        n_positions = len(positions_list)
        position_similarity = np.zeros((n_positions, n_positions))
        
        # Regrouper par position
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
        Unwrap l'environnement pour accéder à l'environnement de base
        """
        base_env = env
        while hasattr(base_env, 'env'):
            base_env = base_env.env
        return base_env
    
    def render_suspicious_positions(self, suspicious_indices = None):
        """
        Affiche les positions et orientations suspectes
        
        Args:
            positions: Liste des positions à afficher (par défaut self.discrete_positions)
            orientations: Liste des orientations à afficher (par défaut [0, 90, 180, 270])
        """
        if suspicious_indices is None:
            print("No suspicious indices provided.")
            return

        if not hasattr(self, 'position_orientation_pairs'):
            print("No position_orientation_pairs found.")
            return

        for idx in suspicious_indices:
            i, j = idx
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
    matrix = np.abs(matrix1 - matrix2)
    below_threshold_indices = []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] > threshold:
                below_threshold_indices.append((i, j))
    print("Indices below threshold:", below_threshold_indices)
    return matrix, below_threshold_indices

if __name__ == '__main__':
    model_path = os.path.abspath('trained_models')
    encoder1 = load_model(model_path=model_path)
    encoder2= resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    feature_dim = 1000
    gym.envs.register(
        id='MyTMaze-v0',
        entry_point='envs.T_maze.custom_T_Maze_V0:MyTmaze'
    )
    
    envs =gym.make("MyTMaze", render_mode='human')
    
   

    
    # Create discretizer
    TmazeforMatrix1 = TmazeDiscretizer(env=envs, encoder=encoder1)
    #TmazeforMatrix2 = TmazeDiscretizer(env=envs, encoder=encoder2, encoder_type='resnet')
    


    # Extract features and compute similarity
    features1 = TmazeforMatrix1.extract_features_from_all_positions()


    matrice1= TmazeforMatrix1.compute_similarity_matrix(features=features1)
    
    '''
    features2 = TmazeforMatrix2.extract_features_from_all_positions()
    matrice2= TmazeforMatrix2.compute_similarity_matrix(features=features2)
    differencematrix, suspicious_indices = difference_matrix(matrice1, matrice2, threshold=1)
    # Visualize
    TmazeforMatrix1.visualize_similarity_matrix(similarity_matrix=differencematrix)
    TmazeforMatrix1.render_suspicious_positions(suspicious_indices=suspicious_indices)

    '''
