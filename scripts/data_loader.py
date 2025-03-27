import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split


def get_dataloader(file_path: str, window_size: int, device, batch_size: int, train_test_split=.0):
    logger = logging.getLogger('dataloader')
    logger.info(f'Data file: {file_path}')
    logger.info(f'Window size: {window_size}')
    logger.info(f'Batch size: {batch_size}')

    df = pd.read_csv(file_path)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)
    label = df['Label'].values
    df.drop(columns='Label', inplace=True)

    windows, labels = [], []
    for i in range(df.shape[0] - window_size + 1):
        window = df[i:i+window_size]
        if window.index.diff().nunique() != 1:  # filter out discontinued windows
            logger.debug('Discontinued window omitted: ({}, {})'.format(
                window.index.min(), window.index.max()))
            continue
        windows.append(window.values)
        labels.append(label[i:i+window_size])
    dataset = TensorDataset(torch.tensor(np.array(windows), dtype=torch.float32, device=device),
                            torch.tensor(np.array(labels), dtype=torch.float32, device=device))

    if train_test_split > 0:
        _test_size = int(len(dataset) * train_test_split)
        _train_size = len(dataset) - _test_size
        _train, _test = random_split(dataset, (_train_size, _test_size))
        return DataLoader(_train, batch_size=batch_size, shuffle=True), DataLoader(_test, batch_size=batch_size, shuffle=False)
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=False), None
