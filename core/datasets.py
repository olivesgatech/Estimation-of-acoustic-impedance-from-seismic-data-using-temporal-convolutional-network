# Set up dataloaders

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import numpy as np

    
class SeismicDataset1D(Dataset):
      """Dataset class for 1D TCN"""
      def __init__(self, seismic, model, trace_indices):
        self.seismic = seismic
        self.model = model
        self.trace_indices = trace_indices
    
    
        assert min(trace_indices) >= 0 and max(trace_indices) + 1 <= len(seismic),"Seismic patch accessing traces out of geometry of the data!"
    
      def __getitem__(self, index):
        trace_index = self.trace_indices[index]
        x = torch.tensor(self.seismic[trace_index][np.newaxis, :], dtype=torch.float).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        y = torch.tensor(self.model[trace_index][np.newaxis, :], dtype=torch.float).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        return x, y
      
      def __len__(self):
        return len(self.trace_indices)     
    
    
