import torch
import math
import mlflow
from mlflow import MlflowClient, MlflowException

from utils.load_standalone_model import load_model
def save_models(models_dict):
    for name in models_dict:
        models_dict[name].to('cpu')
        models_dict[name] = models_dict[name].state_dict()
    
    torch.save(models_dict,'trained_models/saved_from_run.pt')



def create_ml_flow_experiment(experiment_name,uri ="file:mlruns"):
    mlflow.set_tracking_uri(uri)
    try:
        mlflow.set_experiment(experiment_name)
    except MlflowException:
        mlflow.create_experiment(experiment_name)

def get_wall_states(env):
    pos_list = [[1.37*(2*x+1)-0.22, 1.37, -1.37] for x in range(3)] \
                + [[1.37*(2*x+1)-0.22, 1.37, 1.37] for x in range(3)] \
                + [[10.74, 1.37, 1.37*(2*x+1)-6.85] for x in range(5)] \
                + [[8, 1.37, 1.37*(2*x+1)-6.85] for x in [0, 1, 3, 4]] \
                + [[9.37, 1.37, -6.85], [9.37, 1.37, 6.85]]
        
    dir_list = [-math.pi / 2 for _ in range(3)] \
                + [math.pi / 2 for _ in range(3)] \
                + [-math.pi for _ in range(5)] \
                + [0 for _ in range(4)] \
                + [-math.pi / 2 , math.pi / 2]
    
    for p, d in zip(pos_list, dir_list):
        return
              





def collect_features(env, model_path, device, all_layers = False):
    encoder = load_model(model_path= model_path).eval()
    encoder.to(device)

    if device.type == 'mps':
        encoder.compile(backend="aot_eager")
    else:
        encoder.compile()

    for param in encoder.parameters():
        param.requires_grad = False


    states = get_wall_states(env)

    features = encoder(states)

    return features


    
    




    

