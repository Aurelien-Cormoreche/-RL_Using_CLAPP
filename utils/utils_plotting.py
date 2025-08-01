import numpy as np
import matplotlib.pyplot as plt
import torch

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
    features = torch.from_numpy(np.load(file_features)).to('mps')
    model = torch.load(file_model, weights_only= False, map_location=torch.device('mps'))[model_name]
    weights = model['layer.weight'][0]
    
    cos_sim = (features @ weights.T).cpu()
    
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



if __name__ == '__main__':

    tab = [1,100,300,500]
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
        baseline_2  = compute_moving_average('/Volumes/lcncluster/cormorec/rl_with_clapp/mlruns/910472378570111075/2276c172edcc448b92c12ecaec4973e4/metrics/length_episode', t)
        baseline_resnet  = compute_moving_average('/Volumes/lcncluster/cormorec/rl_with_clapp/mlruns/910472378570111075/81476deac679452a9791f811067fda11/metrics/length_episode', t)
        baseline_clapp_color = compute_moving_average('/Volumes/lcncluster/cormorec/rl_with_clapp/mlruns/910472378570111075/614ea0284ea14131b1e8b93400846bd5/metrics/length_episode', t)

        ini_target = compute_moving_average('/Volumes/lcncluster/cormorec/rl_with_clapp/mlruns/910472378570111075/631652bc4453465db86e3476449484d1/metrics/length_episode', t)
        ini_no_target = compute_moving_average('/Volumes/lcncluster/cormorec/rl_with_clapp/mlruns/910472378570111075/c64f82c8acdb4bd8b758891bb190328d/metrics/length_episode', t)

        #plt.plot(baseline_2)
        plt.plot(baseline_resnet)
        plt.plot(ini_target)
        plt.plot(ini_no_target)
        plt.show()

    #print(count_steps(3,'/Volumes/lcncluster/cormorec/rl_with_clapp/mlruns/910472378570111075/d889e0f834f04ed6973d6db00e43635a/metrics/length_episode'))

 
    #visualize_weights('trained_models/saved_from_run.pt', 'critic')
    #meusureIntensityAtPositions('trained_models/encoded_features_CLAPP.npy', '/Volumes/lcncluster/cormorec/rl_with_clapp/trained_models/long_run_1.pt', 'critic')


    #plot_matrix('trained_models/encoded_features_CLAPP.npy')

 