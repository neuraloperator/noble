import torch
from torch.utils.data import Dataset
import numpy as np

class CustomDataset(Dataset):
    """
    A custom PyTorch Dataset for handling input-output pairs with identifiers.
    
    This dataset wraps input tensors, output tensors, and corresponding arrays of identifiers
    to provide a way to train NOBLE with both numeric and string data. 
    
    Args:
        inputs (torch.Tensor): Input data tensor of shape (N, T) where N is the number of samples and T is the sequence length
        outputs (torch.Tensor): Output/target data tensor of shape (N, T) where N is the number of samples and T is the sequence length
        identifiers (np.ndarray of strings): Array of strings identifiers for each sample, used for tracking and debugging purposes
    
    Attributes:
        inputs (torch.Tensor): The input data tensor
        outputs (torch.Tensor): The output/target data tensor  
        identifiers (np.ndarray of strings): Neuron identifiers
    """
    def __init__(self, inputs: torch.Tensor, outputs: torch.Tensor, identifiers: np.ndarray):
        self.inputs = inputs
        self.outputs = outputs
        self.identifiers = identifiers

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx: int):
        return self.inputs[idx], self.outputs[idx], self.identifiers[idx]

def custom_collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor, np.ndarray]]) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """
    Custom collate function for batching data from CustomDataset.
    
    This function takes a batch of tuples containing (input, output, identifier) pairs
    and stacks them into batched tensors and arrays suitable for training NOBLE.

    Args:
        batch (list[tuple[torch.Tensor, torch.Tensor, np.ndarray]]): List of tuples where each tuple
            contains (input_tensor, output_tensor, identifier_arr) for a single sample.
            Input and output tensors should be 1D with shape (T,) where T is the sequence length.
            Identifiers should be numpy arrays containing string identifiers.
    
    Returns:
        tuple[torch.Tensor, torch.Tensor, np.ndarray]: A tuple containing:
            - Batched input tensor of shape (batch_size, T)
            - Batched output tensor of shape (batch_size, T) 
            - Batched numpy array of string identifiers with shape (batch_size,)
    """
    inputs, outputs, identifiers = zip(*batch)
    return torch.stack(inputs), torch.stack(outputs), np.array(identifiers)

class WindowedDataset(Dataset):
    """
    A PyTorch Dataset that creates windowed views of sequences from a base dataset.
    
    This dataset wraps another dataset and extracts fixed-size windows from the input
    and output sequences. It supports both random window selection (for training) and
    fixed window selection (for testing), with configurable start positions and window sizes.
    
    Args:
        base_dataset (Dataset): The base dataset to window. Should return (input, output, identifier)
                               tuples where input and output are 1D tensors and identifier is an array
                               of strings
        window_size (int): Size of the window to extract from sequences
        max_start (int): Maximum allowed start position for window extraction
        take_random_window (bool, optional): If True, randomly select start position for each sample.
                                             If False, use fixed_start. Defaults to True.
        fixed_start (int, optional): Fixed start position to use when take_random_window=False.
                                     Must be in [0, max_start]. Defaults to 0.
    
    Attributes:
        base (Dataset): The underlying base dataset
        window_size (int): Size of the window to extract
        max_start (int): Maximum allowed start position
        random (bool): Whether to use random window selection
        fixed_start (int): Fixed start position for non-random selection
    """

    def __init__(self,
                 base_dataset: Dataset,
                 window_size: int,
                 max_start: int,
                 take_random_window: bool = True,
                 fixed_start: int = 0):

        if max_start < 0:
            raise ValueError("max_start must be non-negative")

        if not (0 <= fixed_start <= max_start):
            raise ValueError("fixed_start must lie in [0, max_start]")

        self.base         = base_dataset
        self.window_size  = window_size
        self.max_start    = max_start
        self.random       = take_random_window
        self.fixed_start  = fixed_start

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        inp_series, out_series, identifier_series = self.base[idx]
        T = inp_series.shape[0]

        if T <= self.window_size:
            start = 0
        else:
            if self.random:
                max_allowed = min(self.max_start, T - self.window_size)
                start = torch.randint(0, max_allowed + 1, ()).item()
            else:
                start = min(self.fixed_start, T - self.window_size)

        inp_win = inp_series[start : start + self.window_size]
        out_win = out_series[start : start + self.window_size]
        
        return inp_win, out_win, identifier_series
