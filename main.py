#!/usr/bin/env python3
import sys
import os
import time
import argparse
import logging
import torch
import torch.nn as nn
import pandas as pd
from models.rnn_clf import Classifier
from models.cnn_gan import Generator, Discriminator
# from models.dnn_gan import Generator, Discriminator
# from models.rnn_gan import Generator, Discriminator
from scripts.gan import train_clf, train_gan
from scripts.data_loader import get_dataloader
from sklearn.preprocessing import MinMaxScaler

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-w', '--window', type=int, default=5)

    parser.add_argument('--epoch-num', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--block-size', type=int, default=10)
    parser.add_argument('--lr-clf', type=float, default=0.001)

    parser.add_argument('--epoch-num-gan', type=int, default=100)
    parser.add_argument('--batch-size-gan', type=int, default=64)
    parser.add_argument('--block-size-gan', type=int, default=10)
    parser.add_argument('--lr-g', type=float, default=0.002)
    parser.add_argument('--lr-d', type=float, default=0.002)

    return parser.parse_args()

def configure_logging(debug=False):
    format = "%(asctime)s - [%(levelname)s] [%(name)s] %(message)s"
    current_time = time.asctime()
    logging_file = '{}/logs/{}.out'.format(
        os.path.dirname(__file__), current_time.replace(' ', '_'))
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(logging_file)
    ]
    os.makedirs(os.path.dirname(logging_file), exist_ok=True)

    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format=format,
        handlers=handlers
    )

def get_dataset(file_path):
    df = pd.read_csv(file_path)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)
    label = df['Label'].values
    df.drop(columns='Label', inplace=True)

    return df, label

def main():
    args = parse_args()
    configure_logging(debug=args.verbose)

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    scaler = MinMaxScaler()
    train, train_label = get_dataset('data/ue_jamming_detection/train.csv')
    valid, valid_label = get_dataset('data/ue_jamming_detection/valid.csv')
    train[train.columns] = scaler.fit_transform(train)
    valid[valid.columns] = scaler.transform(valid)

    train_loader, test_loader = get_dataloader(train, train_label, window_size=args.window, device=device, batch_size=args.batch_size, train_test_split=.2)

    # clf = Classifier().to(device)
    # train_clf(clf, train_loader, test_loader, **args)

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    train_gan(generator, discriminator, train_loader, test_loader, device, **args)

    


if __name__ == "__main__":
    main()
