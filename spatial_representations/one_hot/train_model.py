import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from dataset.T_maze_CLAPP_one_hot.dataset_one_hot import Dataset_One_Hot
from spatial_representations.models import Spatial_Model
from torch.optim import AdamW
from utils.utils_torch import CosineAnnealingWarmupLr
from utils.utils import create_ml_flow_experiment, select_device, parsing
from torch.nn import CrossEntropyLoss
import mlflow
import torch.nn.functional as F
import tqdm

def train_offline(device):

    validation_share = 0.1
    batch_size_training = 32
    batch_size_validation = 32
    num_epochs = 5000
    input_dim = 1024
    hidden_dim = 32
    output_dim = 64
    lr = 1e-3
    warmup_steps = 10
    
    create_ml_flow_experiment('one_hot_training_supervised')
    mlflow.start_run()
    mlflow.log_params(
        {
            'lr' : lr,
            'warmup_steps' : warmup_steps,
            'batch_size_training' : batch_size_training,
            'validation_share' : validation_share,
            'hidden_dim' : hidden_dim,
            'output_dim' : output_dim,
            'num_epochs' : num_epochs
            }
    )

    dataset = Dataset_One_Hot('dataset/T_maze_CLAPP_one_hot/features.pt','dataset/T_maze_CLAPP_one_hot/labels.pt',device= device)
    train_dataset, validation_dataset = random_split(dataset,[1-validation_share, validation_share])
    train_loader = DataLoader(train_dataset, batch_size_training, shuffle= True, pin_memory= True)
    validation_loader = DataLoader(validation_dataset, batch_size_validation, shuffle= False, pin_memory= True)
    
    model = Spatial_Model(input_dim, [output_dim]).to(device)

    optimzer = AdamW(model.parameters(), lr)
    schedulder = CosineAnnealingWarmupLr(optimzer, warmup_steps, num_epochs)

    loss_fn = CrossEntropyLoss(reduction= 'mean')
    
    for epoch in tqdm.tqdm(range(num_epochs)):
        train_loss, train_accuracy = train_one_epoch(train_loader, optimzer, model, loss_fn)
        validation_loss, validation_accuracy = compute_validation_metrics(validation_loader, model, loss_fn)
        log_metrics(train_loss, train_accuracy, validation_loss, validation_accuracy, epoch)
        schedulder.step()

    

def train_one_epoch(train_loader, optimzer, model, loss_fn):
    model.train()
    tot_train_loss = 0
    tot_accuracy = 0
    num_samples = 0
    for i, data in enumerate(train_loader):
        features, labels = data

        optimzer.zero_grad()
        outputs = model(features)
        loss = loss_fn(outputs, labels.squeeze_())
        loss.backward()
        optimzer.step()
        
        tot_train_loss += loss.item()
        predicted = outputs.argmax(dim = 1)
        tot_accuracy += (predicted == labels).sum()

        num_samples += len(labels)
   

    return tot_train_loss/ num_samples, tot_accuracy/ num_samples

def compute_validation_metrics(validation_loader, model, loss_fn):
    model.eval()
    tot_validation_loss = 0
    tot_accuracy = 0
    num_samples = 0
    with torch.no_grad():
        for i, data in enumerate(validation_loader):
            features, labels = data
            outputs = model(features)
            loss = loss_fn(outputs, labels.squeeze_())
            tot_validation_loss += loss.item()
            predicted = outputs.argmax(dim = 1)
            tot_accuracy += (predicted == labels).sum()

            num_samples += len(labels)

    return tot_validation_loss/ num_samples, tot_accuracy/ num_samples
            

def log_metrics(train_loss, train_accuracy, validation_loss, validation_accuracy, epoch):
     mlflow.log_metrics(
         {
             'train_loss' : train_loss,
             'train_accuracy' : train_accuracy,
             'validation_loss' : validation_loss,
             'validation_accuracy' : validation_accuracy
         },
         step= epoch
     )

if __name__ == '__main__':
    args = parsing()
    device = select_device(args)
    train_offline(device)