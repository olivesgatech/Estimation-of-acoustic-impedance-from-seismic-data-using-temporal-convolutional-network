# This script trains the model defined in model file on the marmousi post-stack seismic gathers
import argparse

import torch
import numpy as np
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from core.utils import *
from core.data_loader import *
from core.model import *
from core.results import *

# Fix the random seeds
#torch.backends.cudnn.deterministic = True
#if torch.cuda.is_available(): torch.cuda.manual_seed_all(2019)
#np.random.seed(seed=2019)


# Define function to perform train-val split
def train_val_split(args):
    """Splits dataset into training and validation based on the number of well-logs specified by the user.

    The training traces are sampled uniformly along the length of the model. The validation data is all of the
    AI model except the training traces. Mean and Standard deviation are computed on the training data and used to
    standardize both the training and validation datasets.
    """
    # Load data
    seismic_offsets = marmousi_seismic().squeeze()[:, 100:600]  # dim= No_of_gathers x trace_length
    impedance = marmousi_model().T[:, 400:2400]  # dim = No_of_traces x trace_length

    # Split into train and val
    train_indices = np.linspace(452, 2399, args.n_wells).astype(int)
    val_indices = np.setdiff1d(np.arange(452, 2399).astype(int), train_indices)
    x_train, y_train = seismic_offsets[train_indices], impedance[train_indices]
    x_val, y_val = seismic_offsets[val_indices], impedance[val_indices]

    # Standardize features and targets
    x_train_norm, y_train_norm = (x_train - x_train.mean()) / x_train.std(), (y_train - y_train.mean()) / y_train.std()
    x_val_norm, y_val_norm = (x_val - x_train.mean()) / x_train.std(), (y_val - y_train.mean()) / y_train.std()
    seismic_offsets = (seismic_offsets - x_train.mean()) / x_train.std()

    return x_train_norm, y_train_norm, x_val_norm, y_val_norm, seismic_offsets


# Define train function
def train(args):
    """Sets up the model to train"""
    # Create a writer object to log events during training
    writer = SummaryWriter(pjoin('runs', 'exp_1'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load splits
    x_train, y_train, x_val, y_val, seismic = train_val_split(args)

    # Convert to torch tensors in the form (N, C, L)
    x_train = torch.from_numpy(np.expand_dims(x_train, 1)).float().to(device)
    y_train = torch.from_numpy(np.expand_dims(y_train, 1)).float().to(device)
    x_val = torch.from_numpy(np.expand_dims(x_val, 1)).float().to(device)
    y_val = torch.from_numpy(np.expand_dims(y_val, 1)).float().to(device)
    seismic = torch.from_numpy(np.expand_dims(seismic, 1)).float().to(device)

    # Set up the dataloader for training dataset
    dataset = SeismicLoader(x_train, y_train)
    train_loader = DataLoader(dataset=dataset,
                              batch_size=args.batch_size,
                              shuffle=False)

    # import tcn
    model = TCN(1,
                1,
                args.tcn_layer_channels,
                args.kernel_size,
                args.dropout).to(device)

    #model = ANN().to(device)

    #model = lstm().to(device)

    # Set up loss
    criterion = torch.nn.MSELoss()

    # Define Optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 weight_decay=args.weight_decay,
                                 lr=args.lr)

    # Set up list to store the losses
    train_loss = [np.inf]
    val_loss = [np.inf]
    iter = 0
    # Start training
    for epoch in range(args.n_epoch):
        for x, y in train_loader:
            model.train()
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            writer.add_scalar(tag='Training Loss', scalar_value=loss.item(), global_step=iter)
            if epoch % 200 == 0:
                with torch.no_grad():
                    model.eval()
                    y_pred = model(x_val)
                    loss = criterion(y_pred, y_val)
                    val_loss.append(loss.item())
                    writer.add_scalar(tag='Validation Loss', scalar_value=loss.item(), global_step=iter)
            print('epoch:{} - Training loss: {:0.4f} | Validation loss: {:0.4f}'.format(epoch,
                                                                                        train_loss[-1],
                                                                                        val_loss[-1]))

            if epoch % 100 == 0:
                with torch.no_grad():
                    model.eval()
                    AI_inv = model(seismic)
                fig, ax = plt.subplots()
                ax.imshow(AI_inv[:, 0].detach().cpu().numpy().squeeze().T, cmap="rainbow")
                ax.set_aspect(4)
                writer.add_figure('Inverted Acoustic Impedance', fig, iter)
        iter += 1

    writer.close()

    # Set up directory to save results
    results_directory = 'results'
    seismic_offsets = np.expand_dims(marmousi_seismic().squeeze()[:, 100:600], 1)
    seismic_offsets = torch.from_numpy((seismic_offsets - seismic_offsets.mean()) / seismic_offsets.std()).float()
    with torch.no_grad():
        model.cpu()
        model.eval()
        AI_inv = model(seismic_offsets)

    if not os.path.exists(results_directory):  # Make results directory if it doesn't already exist
        os.mkdir(results_directory)
        print('Saving results...')
    else:
        print('Saving results...')

    np.save(pjoin(results_directory, 'AI.npy'), marmousi_model().T[452:2399, 400:2400])
    np.save(pjoin(results_directory, 'AI_inv.npy'), AI_inv.detach().numpy().squeeze()[452:2399])
    print('Results successfully saved.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=1000,
                        help='# of the epochs. Default = 1000')
    parser.add_argument('--batch_size', nargs='?', type=int, default=10,
                        help='Batch size. Default = 1.')
    parser.add_argument('--tcn_layer_channels', nargs='+', type=int, default=[3, 5, 5, 5, 6, 6, 6, 6],
                        help='No of channels in each temporal block of the tcn. Default = numbers reported in paper')
    parser.add_argument('--kernel_size', nargs='?', type=int, default=5,
                        help='kernel size for the tcn. Default = 5')
    parser.add_argument('--dropout', nargs='?', type=float, default=0.2,
                        help='Dropout for the tcn. Default = 0.2')
    parser.add_argument('--n_wells', nargs='?', type=int, default=10,
                        help='# of well-logs used for training. Default = 19')
    parser.add_argument('--lr', nargs='?', type=float, default=0.001,
                        help='learning rate parameter for the adam optimizer. Default = 0.001')
    parser.add_argument('--weight_decay', nargs='?', type=float, default=0.0001,
                        help='weight decay parameter for the adam optimizer. Default = 0.0001')

    args = parser.parse_args()
    train(args)
    evaluate(args)