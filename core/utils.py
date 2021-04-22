# contains utility functions
import zipfile
import numpy as np

def extract(source_path, destination_path):
    """function extracts all files from a zip file at a given source path
    to the provided destination path"""
    
    with zipfile.ZipFile(source_path, 'r') as zip_ref:
        zip_ref.extractall(destination_path)
    
    
def standardize(seismic, model, no_wells):
    """function standardizes data using statistics extracted from training 
    wells
    
    Parameters
    ----------
    seismic : array_like, shape(num_traces, depth samples)
        2-D array containing seismic section
        
    model : array_like, shape(num_wells, depth samples)
        2-D array containing model section
        
    no_wells : int,
        no of wells (and corresponding seismic traces) to extract.
    """
        
    
    seismic_normalized = (seismic - seismic.mean())/ seismic.std()
    train_indices = (np.linspace(0, len(model)-1, no_wells, dtype=np.int))
    model_normalized = (model - model[train_indices].mean()) / model[train_indices].std()
    
    return seismic_normalized, model_normalized