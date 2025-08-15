import numpy as np
import matplotlib.pyplot as plt
import torch
from RL_algorithms.models import CriticModel, ActorModel
from utils.dimensionality_reduction import PCA
from RL_algorithms.dynamic_encoders import Encoding_Layer
orientations = [0, 45, 90, 135, 180, 225, 270, 315]
positions = [[1.37, 0, 0], [1.37, 0, 0], [4.11, 0, 0], [4.11, 0, 0], [6.8500000000000005, 0, 0], 
                 [6.8500000000000005, 0, 0], [9.78, 0, -5.4799999999999995], [9.78, 0, -2.7399999999999993], 
                 [9.78, 0, 8.881784197001252e-16], [9.78, 0, 2.74], [9.78, 0, 5.480000000000002], [8.5, 0, -5.4799999999999995], 
                 [8.5, 0, -2.7399999999999993], [8.5, 0, 2.74], [8.5, 0, 5.480000000000002], [9.78, 0, -6.0], [9.78, 0, 6.0]]


def load_file(filepath):
    data = np.loadtxt(filepath)
    return data[:, 1]

def compute_moving_average(filepath,window_size, remove_outliers = False, outliers_level = 600):
    data = load_file(filepath)
    if remove_outliers:
        data = data[data <= outliers_level]
    return np.convolve(data, np.ones(window_size)/window_size, mode= 'valid')

def visualize_weights(filepath, model_name):
    dicts = torch.load(filepath, weights_only= False)
    model_dict = dicts[model_name]
    print(model_dict['layer.weight'].shape)
    plt.plot(model_dict['layer.weight'][0].cpu())
    plt.show()

def plot_matrix(file_features):
    features = torch.from_numpy(np.load(file_features)).to('mps') 
    ln1 = torch.nn.LayerNorm((features.shape[1]), elementwise_affine= False).to('mps')
    transformedfeatures = ln1(features)
    cosine = transformedfeatures @ transformedfeatures.T
    plt.matshow(cosine.to('cpu').detach().numpy())
    plt.colorbar()
    plt.show()
    
def meusureIntensityAtPositions(file_features, file_model, model_name):
    features = torch.from_numpy(np.load(file_features))
    model_weights = torch.load(file_model, weights_only= False, map_location='cpu')[model_name]
   
    if model_name == 'critic':
        model = CriticModel(1024)
       
    if model_name == 'actor':
        model = ActorModel(1024,3)
    model.load_state_dict(model_weights)
    model.requires_grad_(False)
   
    cos_sim = model(features)[: ,1]
    
    value_dict = {
    (pos[0], pos[2], ori): cos_sim[i * len(orientations) + j]
    for i, pos in enumerate(positions)
    for j, ori in enumerate(orientations)
    }

    # Normalize values for color mapping
    all_values = np.array(list(value_dict.values()))
    vmin, vmax = all_values.min(), all_values.max()

    # Set up plot
    fig, ax = plt.subplots(figsize=(6, 6))
    cmap = plt.cm.viridis


    for x, z, y in positions:
        for ori in orientations:
            angle_rad = np.deg2rad(ori)
            dx = np.cos(angle_rad)
            dy = np.sin(angle_rad)
           
            val = value_dict[(x, y, ori)]
            color = cmap((val - vmin) / (vmax - vmin))  # Normalize for colormap

            # Draw arrow
            ax.arrow(x, -y, 0.2 * dx, 0.2 * dy, head_width=0.05, color=color)

    # Set plot limits and aspect
    ax.set_aspect('equal')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Orientation Heatmap at Positions")

    # Add colorbar
    from matplotlib.cm import ScalarMappable
    sm = ScalarMappable(cmap=cmap)
    sm.set_array(all_values)
    plt.colorbar(sm, label='Value', ax= ax)

    plt.grid(True)
    plt.show()

def count_steps(frameskip_num, file):
    data = load_file(file)
    return np.sum(data)/frameskip_num

def reduce_data_for_layers(filename_features, filename_labels, num_samples, method, delimiter, model, direction):
    data_features = torch.load(filename_features)[:num_samples]
    data_labels = torch.load(filename_labels)[:num_samples]
    #model.layer[-1] = torch.nn.Identity()
    if direction:
        directions = data_labels % 4
        data_features = torch.cat((data_features, directions), dim = -1)
    with torch.no_grad():
        encoded_features = model(data_features)
    if method == 'PCA':
        pca = PCA(128, 3)
        pca.fit(encoded_features)
        points = pca(encoded_features)
    if delimiter == 'direction_space':
        box = data_labels
    elif delimiter == 'direction':
        box = data_labels % 4
    elif delimiter == 'space':
        box = data_labels // 4
    elif delimiter == 'corridor':
        n = torch.full_like(data_labels, 2)
        n[ data_labels <= 11] = 0
        n[ (data_labels>= 20) & (data_labels <= 23)] = 1
        box = n
    elif delimiter == 'path':
        path_labels = torch.tensor([3,7,8,11,20,23,23,27,28])
        n = torch.full_like(data_labels, 0)
        mask = torch.isin(data_labels, path_labels)
        n[mask] = 1
        box = n
    return points, box

def plot_different_classes(points, box):
    m = torch.max(box) + 1
    for i in torch.arange(m):
        plot_reduced_dimension(points[(box == i).squeeze()], box[box == i])

def plot_reduced_dimension(points, colors):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot with colors per class
    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, cmap='tab10', s=40, alpha=0.8)

    # Label axes
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title('3D PCA Projection of Activations')

    # Optional: add legend using unique class labels
    from matplotlib.lines import Line2D
    unique_labels = np.unique(colors)
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=f'Class {label}',
            markerfacecolor=plt.cm.tab10(label / max(unique_labels)), markersize=10)
        for label in unique_labels
    ]
    ax.legend(handles=legend_elements, loc='best')

    plt.show()

def get_distance_vs_act_distance(reduction, method, model, filename_features, filename_labels, num_samples, plot, directions):
    if reduction:
        p, box = reduce_data_for_layers(filename_features, filename_labels, num_samples, method, 'space', model, True)
    else:
        data_features = torch.load(filename_features)[:num_samples]
        data_labels = torch.load(filename_labels)[:num_samples]
        if directions:
            directions = data_labels % 4
            data_features = torch.cat((data_features, directions), dim = -1)
        with torch.no_grad():
            p = model(data_features)
        box = data_labels//4
    locs = torch.empty((box.shape[0],2))
    box.squeeze_()
    locs[:, 0] = box
    locs[:, 1] = box
    mask = box > 3
    locs[mask,0] =  3
    mask = box <= 3
    locs[mask,1] = 0
    mask = box > 3
    locs[mask,1] = box[box > 3] - 6
    nlocs = locs.unsqueeze(1)
    locs = locs.unsqueeze(0)
    real_dists = torch.abs(locs[:,:,0] - nlocs[:,:,0]) + torch.abs(locs[:,:,1] - nlocs[:,:,1])
    np = p.unsqueeze(1)
    p = p.unsqueeze(0)
    p_dists = torch.sum(torch.abs(p - np), dim = -1)

    mask_flattening = torch.tril(torch.ones_like(real_dists).bool())
    real_dists = real_dists[mask_flattening].flatten()
    p_dists = p_dists[mask_flattening].flatten()
    unique_real_dist = real_dists.unique()
    avg_pdist_per_real = []
    for r_dist in unique_real_dist:
        mask = real_dists == r_dist
        avg_pdist_per_real.append(p_dists[mask].mean().item())
    if plot:
        plt.plot(avg_pdist_per_real)
        plt.show()
    return avg_pdist_per_real

def plot_evolution_of_diffs(absolute,indexi,indexe):
    models = ['2layers1.pt', '2layers2.pt', '2layers3.pt', '2layers10.pt', '2layers30.pt', '2layers80.pt']
    res = []
    for m in models:
        model = CriticModel(1024, False)
        model.load_state_dict(torch.load(f'/Volumes/lcncluster/cormorec/rl_with_clapp/trained_models/{m}', map_location='cpu')['critic'])
        r = get_distance_vs_act_distance(False, 'PCA', model, 'dataset/T_maze_CLAPP_one_hot/features.pt','dataset/T_maze_CLAPP_one_hot/labels.pt',800, False)
        diff = r[indexe] - r[indexi]
        print(diff)
        if not absolute:
            diff = diff/r[indexi]
        res.append(diff)
    plt.plot(res)
    plt.show()

def compute_cosine_similarity_across_class(filename_features, filename_labels, directions, model, num_samples):
    data_features = torch.load(filename_features)[:num_samples]
    data_labels = torch.load(filename_labels)[:num_samples]
    if directions:
        directions = data_labels % 4 
        data_features = torch.cat((data_features, directions), dim = -1)
    with torch.no_grad():
        p = model(data_features)
    
    num_classes = 32
    cos_sim = [[] for _ in range(num_classes)]
    for i in range(num_classes):
        for j in range(num_classes):
            indicesi = data_labels == i
            indicesj = data_labels == j
            fsi = p[indicesi.squeeze()]
            fsj = p[indicesj.squeeze()]
            cosine_similarities = torch.cosine_similarity(fsi, fsj.unsqueeze(1), dim = -1)
            cos_sim[i].append(cosine_similarities.mean())
    
    plt.matshow(cos_sim)
    plt.colorbar()
    plt.show()  
    

def plot_runs():


    tab = [1,50, 100,300,500]
    for t in tab:
        '''
        mv_avg_CLAPP = compute_moving_average('mlruns/244787145723528822/e677b4afb3e349e48481f15f21970daf/metrics/run length', t)
        mv_avg_Resnet = compute_moving_average('mlruns/873129205249233078/08d90e56b9d84e019e5ccee9e9ecc254/metrics/run length',t)
        baselinePPO = compute_moving_average('/Volumes/lcncluster/cormorec/rl_with_clapp/mlruns/332671571023767635/295debefa8ca4e19866a0d75fc055ba2/metrics/run length', t)
 
        #longtrain = compute_moving_average('/Volumes/lcncluster/cormorec/rl_with_clapp/mlruns/910472378570111075/18d5c9c052354170864156ed7bb385fb/metrics/length_episode', t)
        longtrain2 = compute_moving_average('/Volumes/lcncluster/cormorec/rl_with_clapp/mlruns/910472378570111075/d889e0f834f04ed6973d6db00e43635a/metrics/length_episode', t)

        longtrainresnet = compute_moving_average('/Volumes/lcncluster/cormorec/rl_with_clapp/mlruns/910472378570111075/8ff78d03c85d410b839ef10817f9017c/metrics/length_episode', t)
        longtrain3 = compute_moving_average('/Volumes/lcncluster/cormorec/rl_with_clapp/mlruns/910472378570111075/38beab3edf5b476987506eeb90a0f260/metrics/length_episode', t)
        longtrain4 = compute_moving_average('/Volumes/lcncluster/cormorec/rl_with_clapp/mlruns/910472378570111075/f77a81c69a5b461c81b62532445bdfc0/metrics/length_episode', t)
        longtrain5 = compute_moving_average('/Volumes/lcncluster/cormorec/rl_with_clapp/mlruns/910472378570111075/5ddd53ad80d745929b0c1c83a1bc67eb/metrics/length_episode', t)

        train_baseline = compute_moving_average('/Volumes/lcncluster/cormorec/rl_with_clapp/mlruns/910472378570111075/5605406ba03648778118105a0b800018/metrics/length_episode', t)
        train_decay_real = compute_moving_average('/Volumes/lcncluster/cormorec/rl_with_clapp/mlruns/910472378570111075/4aec231385604a358640ffbb85b34876/metrics/length_episode', t)
        '''
        #baseline_2  = compute_moving_average('/Volumes/lcncluster/cormorec/rl_with_clapp/mlruns/910472378570111075/2276c172edcc448b92c12ecaec4973e4/metrics/length_episode', t)
        #baseline_resnet  = compute_moving_average('/Volumes/lcncluster/cormorec/rl_with_clapp/mlruns/910472378570111075/81476deac679452a9791f811067fda11/metrics/length_episode', t)
        #baseline_clapp_color = compute_moving_average('/Volumes/lcncluster/cormorec/rl_with_clapp/mlruns/910472378570111075/614ea0284ea14131b1e8b93400846bd5/metrics/length_episode', t)

        #ini_target = compute_moving_average('/Volumes/lcncluster/cormorec/rl_with_clapp/mlruns/910472378570111075/631652bc4453465db86e3476449484d1/metrics/length_episode', t)
        #ini_no_target = compute_moving_average('/Volumes/lcncluster/cormorec/rl_with_clapp/mlruns/910472378570111075/c64f82c8acdb4bd8b758891bb190328d/metrics/length_episode', t)
        #good_ac =  compute_moving_average('/Volumes/lcncluster/cormorec/rl_with_clapp/mlruns/508181377465869700/6e353a1dca9c42038043434be8c57f30/metrics/length_episode', t)
        #one_hot_ac = compute_moving_average('mlruns/647803037565373307/802c653dc9504059b3004a5ffc76809b/metrics/length_episode', t)
        '''
        just_bias = compute_moving_average('/Volumes/lcncluster/cormorec/rl_with_clapp/mlruns/509629523065386057/360ac29b422d4b239c9a556bf40993e6/metrics/length_episode', t)  
        random_baseline =  compute_moving_average('mlruns/910444605774049268/8497d5a727224391bd3361e1759a68e6/metrics/length_episode', t)
        '''
        try2layers1 = compute_moving_average('/Volumes/lcncluster/cormorec/rl_with_clapp/mlruns/509629523065386057/f376f623caad440c855f194ca708a13e/metrics/length_episode', t)  
        '''
        comp_clapp1 = compute_moving_average('/Volumes/lcncluster/cormorec/rl_with_clapp/mlruns/376693154063831747/75ed0c8d234d43fea9e01bf0bf085294/metrics/length_episode', t)  
        comp_raw1 = compute_moving_average('/Volumes/lcncluster/cormorec/rl_with_clapp/mlruns/376693154063831747/b9403527cce14e4b924ad61afea8ae58/metrics/length_episode', t)  
        comp_res1 = compute_moving_average('/Volumes/lcncluster/cormorec/rl_with_clapp/mlruns/376693154063831747/004edc693b7f4a2d929b3ecd6b79fa9e/metrics/length_episode', t)  
        '''
        comp_clapp5 = compute_moving_average('/Volumes/lcncluster/cormorec/rl_with_clapp/mlruns/376693154063831747/139af87b5c2544a49187a121364e22f1/metrics/length_episode', t)  
        '''
        comp_raw5 = compute_moving_average('/Volumes/lcncluster/cormorec/rl_with_clapp/mlruns/376693154063831747/496bd3aa35334a428d14d3b6dacd9cdf/metrics/length_episode', t)  

        no_images5 = compute_moving_average('/Volumes/lcncluster/cormorec/rl_with_clapp/mlruns/376693154063831747/b6801b24e6574f2eb31e8a32c29bdafa/metrics/length_episode', t)  
        '''


    
        encoder =  compute_moving_average('/Volumes/lcncluster/cormorec/rl_with_clapp/mlruns/707236896616541753/18af403984ab4e82afdfdee0eb42202f/metrics/length_episode', t)  
        four_rooms =  compute_moving_average('/Volumes/lcncluster/cormorec/rl_with_clapp/mlruns/385310356949451556/8ecf5d647fc64edb8e8a98546a11315f/metrics/length_episode', t)  
        four_rooms_2_layers = compute_moving_average('/Volumes/lcncluster/cormorec/rl_with_clapp/mlruns/385310356949451556/0b7f347f44f44be8bbfe5fd6ae7941b3/metrics/length_episode', t)  
        #plt.plot(baseline_2)
        #plt.plot(baseline_resnet)
        #plt.plot(ini_target)
        #plt.plot(ini_no_target)
        #plt.plot(one_hot_ac)
        #plt.plot(good_ac)
        #plt.plot(try2layers1)
        #plt.plot(random_baseline)
        #plt.plot(comp_clapp1)
        #plt.plot(comp_raw1)
        #plt.plot(random_baseline)
        #plt.plot(comp_raw5)
        #plt.plot(comp_clapp5)
        #plt.plot(no_images5)
        #plt.plot(encoder)

        plt.plot(four_rooms)
        plt.plot(four_rooms_2_layers)

        plt.show()

if __name__ == '__main__':

    plot_runs()
    
    #visualize_weights('trained_models/saved_from_run.pt', 'critic')
    #print(torch.load('/Volumes/lcncluster/cormorec/rl_with_clapp/trained_models/saved_from_run.pt', map_location= 'cpu')['critic'].keys())
    #meusureIntensityAtPositions('trained_models/encoded_features_no_images_CLAPP.npy', '/Volumes/lcncluster/cormorec/rl_with_clapp/trained_models/saved_from_run.pt', 'actor')
    #model = CriticModel(1024,1,two_layers= True)
    #model.load_state_dict(torch.load('/Volumes/lcncluster/cormorec/rl_with_clapp/trained_models/2layerswide.pt', map_location='cpu')['critic'])
    #model = Encoding_Layer(1024, 16)
    #model.load_state_dict(torch.load('trained_models/direction_contrastive_encoder.pt',map_location='cpu'))
    #direction = False

    #p, c = reduce_data_for_layers('dataset/T_maze_CLAPP_one_hot/features.pt','dataset/T_maze_CLAPP_one_hot/labels.pt',600, 'PCA', 'direction', model, direction)
    #plot_reduced_dimension(p, c)
    #l = get_distance_vs_act_distance(False, 'PCA', model, 'dataset/T_maze_CLAPP_one_hot/features.pt','dataset/T_maze_CLAPP_one_hot/labels.pt',1600, True, direction)
    #compute_cosine_similarity_across_class('dataset/T_maze_CLAPP_one_hot/features.pt', 'dataset/T_maze_CLAPP_one_hot/labels.pt', direction, model, 600)
    #plot_evolution_of_diffs(False,0, 1)       
    #plot_matrix('trained_models/encoded_features_CLAPP.npy')
 