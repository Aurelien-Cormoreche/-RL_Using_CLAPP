from dataset.T_maze_CLAPP_one_hot.dataset_one_hot import Dataset_One_Hot
import torch
from spatial_representations.models import Spatial_Model
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
def value_for_misclassified(validation_DataLoader, model):
    model.eval().to('mps')
    res1 = torch.tensor([], dtype= torch.float32, device= 'mps')
    res2 = torch.tensor([], dtype= torch.float32, device= 'mps')
    s = torch.nn.Softmax(dim= -1)
    with torch.no_grad():
        for i, batch in enumerate(validation_DataLoader):
            model.to('mps')
            features, labels = batch
            outputs = model(features)
            outputs = s(outputs)
            predicted = outputs.argmax(dim = -1)
            proba_labels = torch.gather(outputs, 1, labels.to(torch.int64))
            proba_predicted = torch.gather(outputs, 1, predicted.unsqueeze(1))
            wrongs = (predicted != labels.squeeze())
            probawrongs = proba_labels[wrongs]
            proba_preds = proba_predicted[wrongs]
            res1 = torch.cat((res1, probawrongs.squeeze(dim= -1)))
            res2 = torch.cat((res2, proba_preds.squeeze(dim= -1)))
            
        return res1, res2

def show_wrongs(validation_DataLoader, model):
    model.eval().to('mps')
    wrongs_labels = torch.tensor([], dtype= torch.float32, device= 'mps')
    with torch.no_grad():
        for i, batch in enumerate(validation_DataLoader):
            features, labels = batch
            outputs = model(features)
            predicted = outputs.argmax(dim = -1)
            wrongs = (predicted != labels.squeeze())
            wrongs_labels = torch.cat((wrongs_labels, predicted[wrongs]))
            
        return wrongs_labels 
def share_of_well_classified_amongst_top_k(validation_DataLoader, model, k):
    model.eval()
    tot = 0
    num_samples = 0
    with torch.no_grad():
        for i, batch in enumerate(validation_DataLoader):
            model.to('mps')
            features, labels = batch
            outputs = model(features)
            tops = outputs.topk(k, dim = -1).indices
            tot += (tops == labels).any(dim = -1).sum().item()
            num_samples += len(labels)
        return tot, num_samples


if __name__ == '__main__':
    one_hot_model_path = 'spatial_representations/one_hot/model.pt'
    validation_dataset_path = 'spatial_representations/one_hot/validation.pt'

    validation_dataset = torch.load(validation_dataset_path, weights_only= False)
    validation_DataLoader = DataLoader(validation_dataset, batch_size= 32)
    one_hot_model = Spatial_Model(1024, [32])
    one_hot_model.load_state_dict(torch.load(one_hot_model_path))
    '''
    res1, res2 = value_for_misclassified(validation_DataLoader, one_hot_model)
    diffs = res2 - res1
    plt.hist(diffs.cpu().numpy(), bins= 60)
    plt.plot()
    plt.show()
    '''
    def plot_share_of_correct_amongst_top_k():
        l = []
        for i in range(1,5):
            res, size = share_of_well_classified_amongst_top_k(validation_DataLoader, one_hot_model, i)
            l.append(res/size)
        plt.plot(range(1,5),l)
        plt.show()

    def plot_wrongly_classified():
        plt.hist(show_wrongs(validation_DataLoader, one_hot_model).cpu().numpy(), bins= 32)
        plt.show()

    plot_wrongly_classified()







    
    

    

    