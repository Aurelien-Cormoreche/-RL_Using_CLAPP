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
    features = torch.from_numpy(np.load(file_features)).to('mps') * 100

    ln1 = torch.nn.LayerNorm((features.shape[1]), elementwise_affine= False).to('mps')
    transformedfeatures = ln1(features)
    
   

    cosine = transformedfeatures @ transformedfeatures.T

    plt.matshow(cosine.to('cpu').detach().numpy())
    plt.colorbar()
    
    plt.show()
    

def meusureIntensityAtPositions(file_features, file_model, model_name):
    features = torch.from_numpy(np.load(file_features)).to('mps')
    model = torch.load(file_model, weights_only= False, map_location=torch.device('mps'))[model_name]
    weights = model['layer.weight'][1]
    
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


if __name__ == '__main__':

    tab = [1,10,20,50,100,300,500]
    for t in tab:
        '''
        mv_avg_CLAPP = compute_moving_average('mlruns/244787145723528822/e677b4afb3e349e48481f15f21970daf/metrics/run length', t)
        mv_avg_Resnet = compute_moving_average('mlruns/873129205249233078/08d90e56b9d84e019e5ccee9e9ecc254/metrics/run length',t)
        mv_avg_a2c =  compute_moving_average('mlruns/244787145723528822/ef58ac2d07e343989ec5dcf2cde369d2/metrics/length_episode',t)
        mv_avg_a2c_fs =  3 * compute_moving_average('mlruns/244787145723528822/fc50ad63a7a542158ae5073c793bc890/metrics/length_episode',t)
        mv_avg_a2c_fs_mf =  compute_moving_average('mlruns/244787145723528822/4c01db63f6804de39fd9106b7812184a/metrics/length_episode',t)
        mv_avg_a2c_fs_mf_2 =  compute_moving_average('mlruns/244787145723528822/2cd2bba123b046c693459074b4050ca2/metrics/length_episode',t)
        mv_avg_a2c_fs2 =  compute_moving_average('mlruns/244787145723528822/e03457730fcf4c7b976ecffcf0845b8d/metrics/length_episode',t)
        imc = compute_moving_average('mlruns/244787145723528822/62f7233037ed48ad909933993444f90d/metrics/length_episode',t)
        new = compute_moving_average('mlruns/244787145723528822/880dc02e877740e080bb84e2874bf976/metrics/length_episode',t)
        higherLr = compute_moving_average('mlruns/244787145723528822/1bf43f94dba04524b3451f7fb072f61f/metrics/length_episode',t)
        normalized_good_1 = compute_moving_average('/Volumes/lcncluster/cormorec/rl_with_clapp/mlruns/742250363624833332/6c08ec70df4b4a6dbe81a35db822efe5/metrics/length_episode', t)
        normalized_good_2 = compute_moving_average('/Volumes/lcncluster/cormorec/rl_with_clapp/mlruns/742250363624833332/d79efb31f4654bdf89f3348791e1d5f4/metrics/length_episode', t)
        normalized_good_3 = compute_moving_average('/Volumes/lcncluster/cormorec/rl_with_clapp/mlruns/742250363624833332/a4ba2d0dfa4c4164b73773a4bd9422f3/metrics/length_episode', t)

        mean_normalized_good = np.array([normalized_good_1,normalized_good_2,normalized_good_3]).mean(axis= 0)

        highlambda = compute_moving_average('/Volumes/lcncluster/cormorec/rl_with_clapp/mlruns/910472378570111075/a82399406d664aeb96cee572193d36eb/metrics/length_episode', t)
        decayinglambda = compute_moving_average('/Volumes/lcncluster/cormorec/rl_with_clapp/mlruns/910472378570111075/38b439f35bc4486fb4888ea07df8ef56/metrics/length_episode', t)
        decayinglambda_decaying_lr_warmup= compute_moving_average('/Volumes/lcncluster/cormorec/rl_with_clapp/mlruns/910472378570111075/df3d79b191724e0393e02b832144f922/metrics/length_episode', t)
        long_decaying_lamd_warmup = compute_moving_average('/Volumes/lcncluster/cormorec/rl_with_clapp/mlruns/910472378570111075/ce1f156b52274e43b088c162292f3965/metrics/length_episode', t)
        small_lambda =  compute_moving_average('/Volumes/lcncluster/cormorec/rl_with_clapp/mlruns/910472378570111075/dd7eeaa3527c49a08108886edce08a90/metrics/length_episode', t)
        '''
        noentropy = compute_moving_average('/Volumes/lcncluster/cormorec/rl_with_clapp/mlruns/910472378570111075/e78e198e983f45a39c0c04a8ef03172a/metrics/length_episode', t)
        noentropy2 = compute_moving_average('/Volumes/lcncluster/cormorec/rl_with_clapp/mlruns/910472378570111075/ec5dfa9d76ca430a9c9c4b469c364a0f/metrics/length_episode', t)
        entropy = compute_moving_average('/Volumes/lcncluster/cormorec/rl_with_clapp/mlruns/910472378570111075/a31e31a9817940b78e273326e95966c0/metrics/length_episode', t)
        entropy2 = compute_moving_average('/Volumes/lcncluster/cormorec/rl_with_clapp/mlruns/910472378570111075/e53abdc9a3a44311a0ce5327532f0888/metrics/length_episode', t)
        #higherLr2 = compute_moving_average('/Volumes/lcncluster/cormorec/rl_with_clapp/mlruns/910472378570111075/ff2ff7a024c74020836a3313a51ede45/metrics/length_episode', t) 
        #plt.plot(mv_avg_CLAPP)
        #plt.plot(mv_avg_Resnet)
        #plt.plot(mv_avg_a2c_fs)
       
        #plt.plot(mv_avg_a2c_fs_mf)
        #plt.plot(mv_avg_a2c_fs_mf_2)
        #plt.plot(mean_normalized_good)
        #plt.plot(mean_normalized_good)
        #plt.plot(highlambda)
        #plt.plot(decayinglambda)
        #plt.plot(long_decaying_lamd_warmup)
        #plt.plot(small_lambda)
        #plt.plot(noentropy)
        #plt.plot(noentropy2)
        plt.plot(entropy)
        plt.plot(entropy2)
        #plt.plot(higherLr2)
       
        plt.show()


    '''
    #visualize_weights('trained_models/saved_from_run.pt', 'critic')
    meusureIntensityAtPositions('trained_models/encoded_features_CLAPP.npy', '/Volumes/lcncluster/cormorec/rl_with_clapp/trained_models/saved_from_run.pt', 'actor')


    #plot_matrix('trained_models/encoded_features_CLAPP.npy')

    '''