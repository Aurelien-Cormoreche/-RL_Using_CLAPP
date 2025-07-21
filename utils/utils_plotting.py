import numpy as np
import matplotlib.pyplot as plt
import torch
def load_file(filepath):
    data = np.loadtxt(filepath)
    return data[:, 1]

def compute_moving_average(filepath,window_size):
    data = load_file(filepath)

    return np.convolve(data, np.ones(window_size)/window_size, mode= 'valid')

def visualize_weights(filepath, model_name):
    dicts = torch.load(filepath, weights_only= False)
    model_dict = dicts[model_name]

    plt.matshow(model_dict['layer.weight'].cpu())
   
    plt.show()

if __name__ == '__main__':

    
    tab = [1,10,20,50,100,300,500]
    for t in tab:
        mv_avg_CLAPP = compute_moving_average('mlruns/244787145723528822/e677b4afb3e349e48481f15f21970daf/metrics/run length', t)
        #mv_avg_Resnet = compute_moving_average('mlruns/873129205249233078/08d90e56b9d84e019e5ccee9e9ecc254/metrics/run length',t)
        #mv_avg_a2c =  compute_moving_average('mlruns/244787145723528822/ef58ac2d07e343989ec5dcf2cde369d2/metrics/length_episode',t)
        mv_avg_a2c_fs =  3 * compute_moving_average('mlruns/244787145723528822/fc50ad63a7a542158ae5073c793bc890/metrics/length_episode',t)
        mv_avg_a2c_fs_mf =  compute_moving_average('mlruns/244787145723528822/4c01db63f6804de39fd9106b7812184a/metrics/length_episode',t)
        mv_avg_a2c_fs_mf_2 =  compute_moving_average('mlruns/244787145723528822/2cd2bba123b046c693459074b4050ca2/metrics/length_episode',t)
        mv_avg_a2c_fs2 =  compute_moving_average('mlruns/244787145723528822/e03457730fcf4c7b976ecffcf0845b8d/metrics/length_episode',t)
        imc = compute_moving_average('mlruns/244787145723528822/43d1946ebacc4ba5876cd446a51b1909/metrics/length_episode',t)
        new = compute_moving_average('mlruns/244787145723528822/880dc02e877740e080bb84e2874bf976/metrics/length_episode',t)
        #plt.plot(mv_avg_CLAPP)
        #plt.plot(mv_avg_Resnet)
        plt.plot(mv_avg_a2c_fs)
       
        #plt.plot(mv_avg_a2c_fs_mf)
        #plt.plot(mv_avg_a2c_fs_mf_2)
        plt.plot(new)
        #plt.plot(imc)
        plt.show()

    '''
    visualize_weights('trained_models/saved_from_run.pt', 'icm_predictor')
    '''
