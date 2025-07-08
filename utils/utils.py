import torch
import mlflow
from mlflow import MlflowClient, MlflowException

def save_model_and_state(model, optimizer, filename):
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        },
        f'trained_models/{filename}'
    )


def create_ml_flow_experiment(experiment_name,uri ="file:mlruns"):
    mlflow.set_tracking_uri(uri)
    try:
        mlflow.set_experiment(experiment_name)
    except MlflowException:
        mlflow.create_experiment(experiment_name)





    

