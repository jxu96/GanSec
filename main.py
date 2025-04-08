#!/usr/bin/env python3
import sys
import os
import time
import argparse
import logging
import torch
import torch.nn as nn
from models.rnn_clf import Classifier
from models.cnn_gan import Generator, Discriminator
# from models.dnn_gan import Generator, Discriminator
# from models.rnn_gan import Generator, Discriminator
from scripts.gan import train_clf, train_gan


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


def main():
    args = parse_args()
    configure_logging(debug=args.verbose)

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    clf = Classifier().to(device)
    train_clf(clf, device, **args)

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    train_gan(generator, discriminator, device, **args)

    


if __name__ == "__main__":
    main()
