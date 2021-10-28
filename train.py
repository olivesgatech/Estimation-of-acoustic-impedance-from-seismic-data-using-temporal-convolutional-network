
## imports
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from os.path import join
from core.utils import extract, standardize
from core.datasets import SeismicDataset1D
from torch.utils.data import DataLoader
from core.model1D import MustafaNet
from sklearn.metrics import r2_score
import errno
import argparse

    
def preprocess(no_wells, data_flag='seam'):
    """Function initializes data, performs standardization, and train test split
    
    Parameters:
    ----------
    no_wells : int,
        number of evenly spaced wells and seismic samples to be evenly sampled 
        from seismic section.

        
    Returns
    -------
    seismic : array_like, shape(num_traces, depth samples)
        2-D array containing seismic section 
        
    model : array_like, shape(num_wells, depth samples)
        2-D array containing model section 

    """
    
    # get project root directory
    project_root = os.getcwd()
    
    if ~os.path.isdir('data'): # if data directory does not exists then extract
        extract('data.zip', project_root)
        
    if data_flag == 'seam':
        # Load data
        seismic = np.load(join('data','poststack_seam_seismic.npy')).squeeze()[:, 50:]
        seismic = seismic[::2, :]
        
        # Load targets and standardize data
        model = np.load(join('data','seam_elastic_model.npy'))[::3,:,::2][:, :, 50:]
        model = model[:,0,:] * model[:,2,:]
    
    else:
        # Load data
        seismic = np.load(join('data','marmousi_synthetic_seismic.npy')).squeeze()
        model= np.load(join('data', 'marmousi_Ip_model.npy')).squeeze()[::5, ::4]
        
    
    # standardize
    seismic, model = standardize(seismic, model, no_wells)
    
    return seismic, model


def train(**kwargs):
    """Function trains 2-D TCN as specified in the paper"""
    
    # obtain data
    seismic, model = preprocess(kwargs['no_wells'], kwargs['data_flag'])
                                                                           
    
    # specify pseudolog positions for training and validation
    traces_seam_train = np.linspace(0, len(model)-1, kwargs['no_wells'], dtype=int)
    traces_seam_validation = np.linspace(0, len(model)-1, 3, dtype=int)
    
    seam_train_dataset = SeismicDataset1D(seismic, model, traces_seam_train)
    seam_train_loader = DataLoader(seam_train_dataset, batch_size = len(seam_train_dataset))
    
    seam_val_dataset = SeismicDataset1D(seismic, model, traces_seam_validation)
    seam_val_loader = DataLoader(seam_val_dataset, batch_size = len(seam_val_dataset))
    
    
    # define device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set up models
    model_seam = MustafaNet().to(device)
    
    # Set up loss
    criterion = torch.nn.MSELoss()
    
    
    optimizer_seam = torch.optim.Adam(model_seam.parameters(),
                                      weight_decay=0.0001,
                                      lr=0.001)
    
    # start training 
    for epoch in range(kwargs['epochs']):
    
      model_seam.train()
      optimizer_seam.zero_grad()
      
      
      for x,y in seam_train_loader:
        y_pred = model_seam(x)
        loss_train = criterion(y_pred, y) 
    
      for x, y in seam_val_loader:
        model_seam.eval()
        y_pred = model_seam(x)
        val_loss = criterion(y_pred, y)
        
    
      loss_train.backward()
      optimizer_seam.step()
      
      print('Epoch: {} | Train Loss: {:0.4f} | Val Loss: {:0.4f} \
            '.format(epoch, loss_train.item(), val_loss.item()))

    
    # save trained models
    if not os.path.isdir('saved_models'):  # check if directory for saved models exists
        os.mkdir('saved_models')
        
    torch.save(model_seam.state_dict(), 'saved_models/model_seam_1D.pth')

def test(**kwargs):
    """Function tests the trained network on SEAM and Marmousi sections and 
    prints out the results"""
    
    # obtain data
    seismic, model = preprocess(kwargs['no_wells'], kwargs['data_flag'])
                                                                          
    
    # define device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # specify pseudolog positions for testing 
    traces_seam_test = np.arange(len(model), dtype=int)
    
    seam_test_dataset = SeismicDataset1D(seismic, model, traces_seam_test)
    seam_test_loader = DataLoader(seam_test_dataset, batch_size = 8)
    
    # load saved models
    if not os.path.isdir('saved_models'):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), 'saved_models')
        
    # set up models
    model_seam = MustafaNet().to(device)
    model_seam.load_state_dict(torch.load('saved_models/model_seam_1D.pth'))
    
    # infer on SEAM
    print("\nInferring ...")
    x, y = seam_test_dataset[0]  # get a sample
    AI_pred = torch.zeros((len(seam_test_dataset), y.shape[-1])).float().to(device)
    AI_act = torch.zeros((len(seam_test_dataset), y.shape[-1])).float().to(device)
    
    mem = 0
    with torch.no_grad():
        for i, (x,y) in enumerate(seam_test_loader):
          model_seam.eval()
          y_pred  = model_seam(x)
          AI_pred[mem:mem+len(x)] = y_pred.squeeze().data
          AI_act[mem:mem+len(x)] = y.squeeze().data
          mem += len(x)
          del x, y, y_pred
    
    vmin, vmax = AI_act.min(), AI_act.max()

    AI_pred = AI_pred.detach().cpu().numpy()
    AI_act = AI_act.detach().cpu().numpy()
    print('r^2 score: {:0.4f}'.format(r2_score(AI_act.T, AI_pred.T)))
    print('MSE: {:0.4f}'.format(np.sum((AI_pred-AI_act).ravel()**2)/AI_pred.size))
    print('MAE: {:0.4f}'.format(np.sum(np.abs(AI_pred - AI_act)/AI_pred.size)))
    print('MedAE: {:0.4f}'.format(np.median(np.abs(AI_pred - AI_act))))
    
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,12))
    ax1.imshow(AI_pred.T, vmin=vmin, vmax=vmax, extent=(0,35000,15000,0))
    ax1.set_aspect(35/30)
    ax1.set_xlabel('Distance Eastimg (m)')
    ax1.set_ylabel('Depth (m)')
    ax1.set_title('Predicted')
    ax2.imshow(AI_act.T, vmin=vmin, vmax=vmax, extent=(0,35000,15000,0))
    ax2.set_aspect(35/30)
    ax2.set_xlabel('Distance Eastimg (m)')
    ax2.set_ylabel('Depth (m)')
    ax2.set_title('Ground-Truth')
    plt.show()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyperparams')
    
    parser.add_argument('--epochs', nargs='?', type=int, default=900,
                        help='Number of epochs. Default = 1000')
    parser.add_argument('--no_wells', nargs='?', type=int, default=12,
                        help='Number of sampled pseudologs for seismic section. Default = 12.')
    parser.add_argument('--data_flag', type=str, default='seam', choices=['seam', 'marmousi'],
                        help='Data flag to specify the dataset used to train the model')


    args = parser.parse_args()
    
    train(no_wells=args.no_wells, epochs=args.epochs, data_flag=args.data_flag)
    test(no_wells=args.no_wells, epochs=args.epochs, data_flag=args.data_flag)