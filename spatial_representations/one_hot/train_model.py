import torch
import os
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchrl.data import ReplayBuffer, ListStorage
from dataset.T_maze_CLAPP_one_hot.dataset_one_hot import Dataset_One_Hot
from spatial_representations.models import Spatial_Model
from torch.optim import AdamW
from utils.utils_torch import CosineAnnealingWarmupLr
from utils.utils import create_ml_flow_experiment, select_device, parsing, createPCA, create_envs
from utils.load_standalone_model import load_model
from torch.nn import CrossEntropyLoss
import mlflow
import torch.nn.functional as F
import tqdm


import random
def train_online(args):
    num_steps_collect = 64
    num_epochs = 500000
    lr = 1e-6
    input_dim = 1024
    output_dim = 32
    batch_size = 32
    size_replay_buffer = 1000
    num_updates = 5
    create_ml_flow_experiment('one_hot_training_online')
    mlflow.start_run()
    mlflow.log_params(
        {
            'lr' : lr,
            }
    )

    envs = create_envs(args, 1, reward= False)
    state_ini = envs.reset(seed= args.seed)
    state_ini = torch.unsqueeze(state_ini, dim= 1)
    state_ini = state_ini.reshape(state_ini.shape[0], state_ini.shape[3], state_ini.shape[1], state_ini.shape[2])
    model = Spatial_Model(input_dim, [output_dim]).to(device)
    optimzer = AdamW(model.parameters(), lr, weight_decay=0.001 , amsgrad= True)
    encoder = load_model(os.path.abspath('trained_models')).to(device).eval().requires_grad_(False)
    loss_fn = torch.nn.CrossEntropyLoss()

    replay_buffer = ReplayBuffer(storage= ListStorage(max_size= size_replay_buffer), batch_size= batch_size)

    for epoch in range(num_epochs):
        collected = torch.empty((num_steps_collect, state_ini.shape[1], state_ini.shape[2], state_ini.shape[3]), dtype= torch.float32, device= args.device)
        labels_collected = torch.empty((num_steps_collect, 1), dtype= torch.float32, device= device)
        for step in range(num_steps_collect):
            obs = envs.step(random.randint(0,2))[0]
            obs = torch.tensor(obs, dtype= torch.float32, device= args.device)
            obs = obs.reshape(obs.shape[3], obs.shape[1], obs.shape[2])
            collected[step] = obs


        encoded = encoder(collected)
        to_store = torch.vstack((encoded, labels_collected))
        replay_buffer.extend(to_store)
        tot_accuracy = 0
        for upating in range(num_updates):
            
            sampled = replay_buffer.sample()
            features = sampled[:, :-1]
            labels = sampled[:, -1]
            outputs = model(features)
            optimzer.zero_grad()
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimzer.step()
            tot_accuracy += (outputs == labels).sum().item()/len(labels)
        
        mlflow.log_metric('accuracy', tot_accuracy/num_updates, step= epoch)

def train_offline(device):

    validation_share = 0.1
    batch_size_training = 32
    batch_size_validation = 64
    num_epochs = 800
    input_dim = 1024
    hidden_dim = 512
    output_dim = 32
    lr = 5e-5
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
    
    torch.save(validation_dataset, 'spatial_representations/one_hot/validation.pt')

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