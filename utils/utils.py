import torch
import mlflow
from mlflow import MlflowClient, MlflowException

def save_models(models_dict):
    for name in models_dict:
        models_dict[name].to('cpu')
        models_dict[name] = models_dict[name].state_dict()
    
    torch.save(models_dict,'trained_models/saved_from_run')



def create_ml_flow_experiment(experiment_name,uri ="file:mlruns"):
    mlflow.set_tracking_uri(uri)
    try:
        mlflow.set_experiment(experiment_name)
    except MlflowException:
        mlflow.create_experiment(experiment_name)





    

