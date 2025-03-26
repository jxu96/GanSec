#!/usr/bin/env python3
import sys
import os
import time
import argparse
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-w', '--window', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=64)

    return parser.parse_args()

def configure_logging(debug=False):
    format = "%(asctime)s - [%(levelname)s] [%(name)s] %(message)s"
    current_time = time.asctime()
    logging_file = '{}/logs/{}.out'.format(os.path.dirname(__file__), current_time.replace(' ', '_'))
    handlers = [
        logging.StreamHandler(sys.stdout),
        # logging.FileHandler(logging_file)
    ]
    os.makedirs(os.path.dirname(logging_file), exist_ok=True)

    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format=format,
        handlers=handlers
    )

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
        if window.index.diff().nunique() != 1: # filter out discontinued windows
            logger.debug('Discontinued window omitted: ({}, {})'.format(window.index.min(), window.index.max()))
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

def main():
    args = parse_args()
    configure_logging(debug=args.verbose)

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    train, test = get_dataloader('data/ue_jamming_detection/train.csv', window_size=args.window, device=device, batch_size=args.batch_size, train_test_split=.2)
    # valid, _ = get_dataloader('data/ue_jamming_detection/valid.csv', window_size=args.window, device=device, batch_size=args.batch_size)

if __name__ == "__main__":
    main()
