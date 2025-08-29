import os
from RL_algorithms.trainer import Trainer
from RL_algorithms.models import Encoder_Model, Keras_Encoder_Model
from utils.load_standalone_model import load_model
from utils.utils import save_models, create_ml_flow_experiment, parsing, create_envs, launch_experiment, createPCA, select_device
from spatial_representations.models import Spatial_Model
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights, VGG16_Weights, vgg16
from utils.visualize_policy import visualize_policy
from RL_algorithms.run_separate_dynamic_encoder import run_separate_dynamic_encoder
from RL_algorithms.dynamic_encoders import Encoding_Layer, Pretrained_Dynamic_Encoder
import numpy as np
import mlflow
from huggingface_hub import from_pretrained_keras

def train(opt, envs, model_path, device, models_dict):
    """
    Runs a single training or evaluation session with the specified encoder and environment.
    """

    encoder_models = [] # List of encoder components to be stacked/wrapped
    
    # ----- Encoder Selection -----
    if opt.encoder.startswith('CLAPP'):
        #use CLAPP pretrained on STL10
        encoder_models.append(load_model(model_path= model_path).eval())
        feature_dim = 1024
        if opt.keep_patches: #use patch based features (keep the spatial information of patches)
            feature_dim = 15 * 1024
    elif opt.encoder.startswith('resnet'):
        #use resnet50 pretrained on ImangeNet1K 
        transform = ResNet50_Weights.IMAGENET1K_V2.transforms()
        model_res = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        model_res.fc = torch.nn.Identity()
        assert not opt.greyscale #resnet can not be run on grayscale images
        feature_dim = 2048
        encoder_models.append(transform)
        encoder_models.append(model_res)
    elif opt.encoder.startswith('raw'):
        #raw means that there is no encoder, we just train on raw pixels
        feature_dim = 60 * 80
        start_dim_flatten = -2
        if not opt.greyscale:
            feature_dim *= 3
            start_dim_flatten = -3
        encoder_models.append(torch.nn.Flatten(start_dim_flatten))
    elif opt.encoder.startswith('simclr'):
        feature_dim = 128
        model = from_pretrained_keras("keras-io/semi-supervised-classification-simclr").encoder()
        encoder_models.append(Keras_Encoder_Model(model))

    # ----- One-hot encoding option -----
    if opt.encoder.endswith('one_hot'):
        #one hot encoder, pretrained one hot encoder that was trained in a supervised manner, predict the location from features 
        #should be used after another encoder (here it was pretrained after CLAPP)
        one_hot_model = Spatial_Model(feature_dim, [32])
        one_hot_model.load_state_dict(torch.load('spatial_representations/one_hot/model.pt', map_location= device))
        feature_dim = 32
        encoder_models.append(one_hot_model)
        encoder_models.append(nn.Softmax(dim= -1))
        print('using one hot')
    
    # ----- Pretrained temporal encoder option -----
    if opt.encoder_layer == 'pretrained':
        #use of an encoder that was pretrained in order to decorrelate the feature in a place cells like manner
        #encoder was pretrained by having an agent run around and changing representations with a contrastive loss
        encoder_time = Encoding_Layer(feature_dim, opt.encoder_latent_dim_time)
        encoder_time.load_state_dict(torch.load(f'trained_models/{opt.encoder_model_time}', map_location= device))
        pretrained_encoder = Pretrained_Dynamic_Encoder([encoder_time], opt.encoder_output_mode).to(device)
        if opt.encoder_output_mode == 'replace': #if we should concatenate the output with the feautures or just replace the features completely
            feature_dim = opt.encoder_latent_dim_time 
        else:
            feature_dim += opt.encoder_latent_dim_time 
        encoder_models.append(pretrained_encoder)

    #create encoder and freeze it
    encoder = Encoder_Model(encoder_models)
    encoder = encoder.to(device).requires_grad_(False)
    encoder.compile(backend="aot_eager")

    action_dim = envs.single_action_space.n
    feature_dim = feature_dim * opt.nb_stacked_frames

    # ----- Train or Evaluate -----
    if opt.task == 'train':
        #train our agent for navigation in the given environment
        trainer = Trainer(opt, envs, encoder, feature_dim, action_dim)
        trainer.train()
    else:
        #train out extra encoder layer to decorrelate the representations 
        run_separate_dynamic_encoder(opt, envs, encoder, feature_dim, opt.num_epochs, action_dim)

    envs.close()
 


def main(args):
    """
    Main entry point: sets up MLflow, environments, and launches training/evaluation.
    """

   
    device = select_device(args)# Pick GPU if available

    model_path = os.path.abspath('trained_models')

    models_dict = {}
    # Create MLflow experiment for logging
    create_ml_flow_experiment(args.experiment_name)
    

    if args.experiment:
        # Define a set of runs with hyperparameters
        run_dicts = [
            { 'run_name' : 'CLAPP_Normalized_2',
              'algorithm' : 'actor_critic_e',
              'encoder' : 'CLAPP',
              'greyscale' : True,
              'num_epochs' : 8000,
              'frame_skip' : 3,
              'num_envs' : 1,
              'actor_lr' : 5e-5,
              'critic_lr' : 1e-4,
              't_delay_theta' : 0.9,
              't_delay_w' : 0.9,
              'gamma' : 0.995,
              'normalize_features' : True,
            }
,               
        ]
        # Seeds for reproducibility
        seeds = [5,10]
        # Launch experiments across runs and seeds with the parameters given in run_dicts
        launch_experiment(args, run_dicts, seeds,args.experiment_name, device, models_dict)
    else:
        #create miniworld envrionments for training
        envs = create_envs(args, args.num_envs, not args.no_reward)
        # Run a single training session
        train(opt= args, envs= envs,model_path= model_path,device =device, models_dict= models_dict)
    
if __name__ == '__main__':
    args = parsing() # Parse CLI arguments
    main(args) # Run main pipeline
