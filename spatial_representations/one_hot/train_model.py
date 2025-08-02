import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from dataset.T_maze_CLAPP_one_hot.dataset_one_hot import Dataset_One_Hot
from spatial_representations.models import Spatial_Model
from torch.optim import AdamW
from utils.utils_torch import CosineAnnealingWarmupLr
from utils.utils import create_ml_flow_experiment, select_device, parsing, createPCA
from torch.nn import CrossEntropyLoss
import mlflow
import torch.nn.functional as F
import tqdm

def train_offline(device):

    validation_share = 0.1
    batch_size_training = 32
    batch_size_validation = 64
    num_epochs = 800
    input_dim = 1024
    hidden_dim = 512
    output_dim = 32
    lr = 8e-5
    warmup_steps = 30
    n_elements = 4096
    checkpoint_model = 250
    
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
    #pca =createPCA(None, f'dataset/T_maze_CLAPP_one_hot/features.pt', None, None, input_dim, n_elements )
    #t = lambda x : torch.tensor(pca.transform(x.to('cpu').numpy().reshape(1, -1)), device= device)
    dataset = Dataset_One_Hot('dataset/T_maze_CLAPP_one_hot/features.pt','dataset/T_maze_CLAPP_one_hot/labels.pt',device= device, transforms= None)
    train_dataset, validation_dataset = random_split(dataset,[1-validation_share, validation_share])
    train_loader = DataLoader(train_dataset, batch_size_training, shuffle= True, pin_memory= True)
    validation_loader = DataLoader(validation_dataset, batch_size_validation, shuffle= False, pin_memory= True)
    
    model = Spatial_Model(input_dim, [output_dim]).to(device)
    optimzer = AdamW(model.parameters(), lr, weight_decay=0.001 , amsgrad= True)
    #schedulder = CosineAnnealingWarmupLr(optimzer, warmup_steps, num_epochs)

    loss_fn = CrossEntropyLoss(reduction= 'sum')
    
    for epoch in tqdm.tqdm(range(num_epochs)):
        train_loss, train_accuracy = train_one_epoch(train_loader, optimzer, model, loss_fn)
        validation_loss, validation_accuracy = compute_validation_metrics(validation_loader, model, loss_fn)
        log_metrics(train_loss, train_accuracy, validation_loss, validation_accuracy, epoch)
        if epoch % checkpoint_model == 0:
            torch.save(model.state_dict(), 'spatial_representations/one_hot/model.pt')
        #schedulder.step()

    

def train_one_epoch(train_loader, optimzer, model, loss_fn):
    model.train()
    tot_train_loss = 0
    tot_accuracy = 0
    num_samples = 0
    for i, data in enumerate(train_loader):
        features, labels = data
        features = features.squeeze()
        labels = labels.squeeze()
        optimzer.zero_grad()
        outputs = model(features)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimzer.step()
        tot_train_loss += loss.item() 
        predicted = outputs.argmax(dim = 1)
        tot_accuracy += (predicted == labels).sum().item()
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
            features = features.squeeze()
            labels = labels.squeeze()
            outputs = model(features)
            loss = loss_fn(outputs, labels)
            tot_validation_loss += loss.item()
            predicted = outputs.argmax(dim = 1)
            tot_accuracy += (predicted == labels).sum().item()

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