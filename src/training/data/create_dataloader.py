import os, torch, pickle
import numpy as np
from torch.utils.data import DataLoader
from training.data.datasets import CustomDataset, WindowedDataset, custom_collate_fn

def create_dataloader(
    config: dict,
    window_size_ms: float = None,
    train_max_start_ms: float = None,
    test_fixed_start_ms: float = None,
    dt: float = None,
    ds_factor: int = None,
    take_random_window: bool = True,
    window: bool = True) -> tuple[DataLoader, DataLoader]:

    """
    Create DataLoader objects for training and testing NOBLE.
    
    This function loads simulation data from a pickle file and creates PyTorch DataLoader
    objects for training and testing. It supports both full-length sequences and windowed
    sequences.
    
    Args:
        config (dict): Configuration dictionary containing:
            - params.paths.data_path: Path to the pickle file containing simulation data
            - params.training.device: Device to load tensors on ('cpu' or 'cuda')
            - params.training.batch_size_train: Batch size for training DataLoader
            - params.training.batch_size_test: Batch size for testing DataLoader
        window_size_ms (float, optional): Size of the window in milliseconds. Required if window=True.
        train_max_start_ms (float, optional): Maximum start time for training windows in milliseconds. Required if window=True.
        test_fixed_start_ms (float, optional): Fixed start time for test windows in milliseconds. Required if window=True.
        dt (float, optional): Original time step in milliseconds. Required if window=True.
        ds_factor (int, optional): Downsampling factor. Required if window=True.
        take_random_window (bool, optional): Whether to take random windows during training. Defaults to True.
        window (bool, optional): Whether to use windowed datasets. Defaults to True.
    
    Returns:
        tuple[DataLoader, DataLoader]: A tuple containing (train_loader, test_loader) where each loader is a PyTorch DataLoader.
    """

    data_path        = config['params']['paths']['data_path']
    device           = config['params']['training']['device']
    batch_size_train = config['params']['training']['batch_size_train']
    batch_size_test  = config['params']['training']['batch_size_test']

    data_path = os.path.expanduser(data_path)

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    train_input        = torch.tensor(data["train"]["stimulus"], dtype=torch.float32) * 1e10
    train_output       = torch.tensor(data["train"]["response"], dtype=torch.float32)
    train_identifier   = data["train"]["identifier"]

    test_input      = torch.tensor(data["test"]["stimulus"], dtype=torch.float32) * 1e10
    test_output     = torch.tensor(data["test"]["response"], dtype=torch.float32)
    test_identifier = data["test"]["identifier"]

    train_ds = CustomDataset(train_input, train_output, train_identifier)
    test_ds  = CustomDataset(test_input,  test_output, test_identifier)

    if window:
        if None in [window_size_ms, train_max_start_ms, test_fixed_start_ms, dt, ds_factor]:
            raise ValueError("window_size_ms, train_max_start_ms, test_fixed_start_ms, dt, and ds_factor must be provided when window=True.")
        
        sample_interval_ms   = dt * ds_factor
        window_size_pts      = int(window_size_ms / sample_interval_ms)
        train_max_start_pts  = int(train_max_start_ms / sample_interval_ms)
        test_fixed_start_pts = int(test_fixed_start_ms / sample_interval_ms)

        train_ds = WindowedDataset(
            base_dataset       = train_ds,
            window_size        = window_size_pts,
            max_start          = train_max_start_pts,
            take_random_window = take_random_window)

        test_ds = WindowedDataset(
            base_dataset       = test_ds,
            window_size        = window_size_pts,
            max_start          = test_fixed_start_pts,
            take_random_window = False,
            fixed_start        = test_fixed_start_pts)

    train_loader = DataLoader(train_ds, batch_size=batch_size_train, shuffle=True, collate_fn=custom_collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size_test, shuffle=True, collate_fn=custom_collate_fn)
    
    return train_loader, test_loader
