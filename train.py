#!/usr/bin/env python3
import sys
import os
import time
import argparse
import logging
import torch
from scripts.gan import train_ec_gan, train_co_gan, save_gan
from scripts.data_loader import get_dataloader, get_windows, get_dataset
from sklearn.preprocessing import MinMaxScaler
from scipy.io import savemat

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-m', '--model', type=str)
    parser.add_argument('-w', '--window', type=int, default=5)

    # EC-GAN
    parser.add_argument('--ec-epoch-num', type=int, default=100)
    parser.add_argument('--ec-batch-size', type=int, default=64)
    parser.add_argument('--ec-block-size', type=int, default=-1)
    parser.add_argument('--ec-lr-g', type=float, default=0.001)
    parser.add_argument('--ec-lr-d', type=float, default=0.001)

    # CO-GAN
    parser.add_argument('--co-epoch-num', type=int, default=100)
    parser.add_argument('--co-batch-size', type=int, default=64)
    parser.add_argument('--co-block-size', type=int, default=-1)
    parser.add_argument('--co-lr-g', type=float, default=0.001)
    parser.add_argument('--co-lr-d', type=float, default=0.001)

    return parser.parse_args()

def configure_logging(output, debug=False):
    format = "%(asctime)s - [%(levelname)s] [%(name)s] %(message)s"
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(output)
    ]

    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format=format,
        handlers=handlers
    )

def main():
    args = parse_args()
    current_time = time.asctime().replace(' ', '_')
    folder = f'backups/{current_time}_{args.model.upper()}'
    os.makedirs(folder, exist_ok=True)

    configure_logging(output=f'{folder}/train.out', debug=args.verbose)
    logging.info(args)

    if args.model == 'dnn':
        from models.dnn import Generator, Discriminator, LabeledDiscriminator
    elif args.model == 'cnn':
        from models.cnn import Generator, Discriminator, LabeledDiscriminator
    elif args.model == 'rnn':
        from models.rnn import Generator, Discriminator, LabeledDiscriminator

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    logging.info(f'using device: {device}')

    scaler = MinMaxScaler()
    A, A_label = get_dataset('data/ue_jamming_detection/train.csv')
    B, B_label = get_dataset('data/ue_jamming_detection/valid.csv')
    A[A.columns] = scaler.fit_transform(A)
    B[B.columns] = scaler.transform(B)

    A, A_label = get_windows(A, A_label, args.window)
    B, B_label = get_windows(B, B_label, args.window)

    ### Train GAN
    generator = Generator(
        data_shape=[A.shape[1], A.shape[2]],
        label_shape=[A_label.shape[1], 1],
        label_embedding_size=8,
        ).to(device)

    discriminator = Discriminator(
        data_shape=[A.shape[1], A.shape[2]],
        label_shape=[A_label.shape[1], 1],
        label_embedding_size=8,
        ).to(device)
    
    A_train_loader, A_test_loader = get_dataloader(
        A, A_label, device=device, batch_size=args.ec_batch_size, train_test_split=.2)
    
    traces = train_ec_gan(generator, discriminator, A_train_loader, A_test_loader, device, args)
    save_gan(generator, discriminator, loc=f'{folder}/ec-gan')
    savemat(f'{folder}/traces_ec.mat', traces)

    ### Train Labeled GAN
    generator = Generator(
        data_shape=[A.shape[1], A.shape[2]],
        label_shape=[A_label.shape[1], 1],
        label_embedding_size=8,
        ).to(device)

    discriminator = LabeledDiscriminator(
        data_shape=[A.shape[1], A.shape[2]],
        label_shape=[A_label.shape[1], 1],
        ).to(device)

    A_train_loader, A_test_loader = get_dataloader(
        A, A_label, device=device, batch_size=args.co_batch_size, train_test_split=.2)
    
    traces = train_co_gan(generator, discriminator, A_train_loader, A_test_loader, device, args)
    save_gan(generator, discriminator, loc=f'{folder}/co-gan')
    savemat(f'{folder}/traces_co.mat', traces)

if __name__ == "__main__":
    main()
