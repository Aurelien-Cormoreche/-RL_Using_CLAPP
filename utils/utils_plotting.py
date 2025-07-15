import numpy as np
import matplotlib.pyplot as plt

def load_file(filepath):
    data = np.loadtxt(filepath)
    return data[:, 1]

def compute_moving_average(filepath,window_size):
    data = load_file(filepath)

    return np.convolve(data, np.ones(window_size)/window_size, mode= 'valid')



if __name__ == '__main__':
    
    tab = [20,50,100,300,500]
    for t in tab:
        mv_avg_CLAPP = compute_moving_average('mlruns/244787145723528822/e677b4afb3e349e48481f15f21970daf/metrics/run length', t)
        mv_avg_Resnet = compute_moving_average('mlruns/873129205249233078/08d90e56b9d84e019e5ccee9e9ecc254/metrics/run length',t)
        plt.plot(mv_avg_CLAPP)
        plt.plot(mv_avg_Resnet)
        plt.show()